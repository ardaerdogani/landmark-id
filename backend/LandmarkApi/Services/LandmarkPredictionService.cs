using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace LandmarkApi.Services;

public class LandmarkPrediction
{
    public required string Label { get; set; }
    public float Confidence { get; set; }
    public int Rank { get; set; }
    public string? WikipediaUrl { get; set; }
}

public class PredictionResult
{
    public required List<LandmarkPrediction> Predictions { get; set; }
    public long InferenceTimeMs { get; set; }
    public int? PredictionRecordId { get; set; }
}

/// <summary>
/// Landmark prediction service using Python TFLite interpreter via subprocess.
/// </summary>
public class LandmarkPredictionService : IDisposable
{
    // Bump this whenever you change the embedded python to force refresh.
    private const string PythonScriptVersion = "2026-01-03_backend_input_fix_v1";

    private readonly string[] _labels;
    private readonly string _modelPath;
    private readonly string _pythonPath;
    private const int ImageSize = 224;

    private readonly ILogger<LandmarkPredictionService> _logger;
    private readonly WikipediaService _wikipediaService;

    public LandmarkPredictionService(
        ILogger<LandmarkPredictionService> logger,
        IConfiguration configuration,
        WikipediaService wikipediaService)
    {
        _logger = logger;
        _wikipediaService = wikipediaService;

        // IMPORTANT: use config if provided (your appsettings.json points to fp32)
        var modelRel = configuration["ModelPath"] ?? "Models/landmark_mnv3_fp32.tflite";
        var labelsRel = configuration["LabelsPath"] ?? "Models/labels.txt";
        _pythonPath = configuration["PythonPath"] ?? "python";

        static string Resolve(string p) =>
            Path.IsPathRooted(p) ? p : Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, p));

        _modelPath = Resolve(modelRel);
        var labelsPath = Resolve(labelsRel);

        _logger.LogInformation("Model path (resolved): {ModelPath}", _modelPath);
        _logger.LogInformation("Labels path (resolved): {LabelsPath}", labelsPath);

        if (!File.Exists(_modelPath))
            throw new FileNotFoundException($"Model file not found: {modelRel}", _modelPath);

        if (!File.Exists(labelsPath))
            throw new FileNotFoundException($"Labels file not found: {labelsRel}", labelsPath);

        _labels = File.ReadAllLines(labelsPath)
            .Select(l => l.Trim())
            .Where(l => !string.IsNullOrWhiteSpace(l))
            .ToArray();

        _logger.LogInformation("Service initialized. {ClassCount} classes detected", _labels.Length);
    }

    public async Task<PredictionResult> PredictAsync(Stream imageStream)
    {
        var stopwatch = Stopwatch.StartNew();

        // Save image temporarily
        var tempImagePath = Path.Combine(Path.GetTempPath(), $"temp_image_{Guid.NewGuid()}.jpg");

        try
        {
            // Load and preprocess image (force RGB and resize to 224x224)
            using (var image = await Image.LoadAsync<Rgb24>(imageStream))
            {
                image.Mutate(x => x.Resize(ImageSize, ImageSize));
                await image.SaveAsJpegAsync(tempImagePath);
            }

            // Python script path in runtime base dir (same place we execute from)
            var scriptPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "predict_tflite.py");

            // Ensure python script is up-to-date
            var scriptNeedsRefresh = true;
            if (File.Exists(scriptPath))
            {
                try
                {
                    var content = await File.ReadAllTextAsync(scriptPath);
                    scriptNeedsRefresh = !content.Contains(PythonScriptVersion);
                }
                catch
                {
                    scriptNeedsRefresh = true;
                }
            }

            if (scriptNeedsRefresh)
            {
                _logger.LogWarning("Creating/refreshing Python inference script...");
                await CreatePythonScript(scriptPath);
            }

            // Call Python script
            var result = await RunPythonInference(tempImagePath);

            // Enrich with Wikipedia URLs
            foreach (var prediction in result.Predictions)
            {
                prediction.WikipediaUrl = await _wikipediaService.GetWikipediaUrlAsync(prediction.Label);
            }

            stopwatch.Stop();
            result.InferenceTimeMs = stopwatch.ElapsedMilliseconds;

            return result;
        }
        finally
        {
            // Clean up temp file
            if (File.Exists(tempImagePath))
            {
                try { File.Delete(tempImagePath); } catch { /* ignore */ }
            }
        }
    }

    private async Task<PredictionResult> RunPythonInference(string imagePath)
    {
        var scriptPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "predict_tflite.py");

        var psi = new ProcessStartInfo
        {
            FileName = _pythonPath,
            Arguments = $"\"{scriptPath}\" \"{_modelPath}\" \"{imagePath}\" \"{_labels.Length}\"",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        using var process = new Process { StartInfo = psi };
        process.Start();

        var outputTask = process.StandardOutput.ReadToEndAsync();
        var errorTask = process.StandardError.ReadToEndAsync();

        await process.WaitForExitAsync();

        var output = await outputTask;
        var error = await errorTask;

        // Keep stderr visible for debugging (quantization, x_min/x_max, etc.)
        if (!string.IsNullOrWhiteSpace(error))
            _logger.LogInformation("Python stderr: {Stderr}", error.Trim());

        if (process.ExitCode != 0)
            throw new Exception($"Python inference failed (exit {process.ExitCode}): {error}");

        // Parse JSON output from Python script
        var predictions = JsonSerializer.Deserialize<List<PythonPrediction>>(output)
            ?? throw new Exception($"Failed to parse Python output. Raw: {output}");

        var mapped = predictions.Select((p, idx) =>
        {
            var safeIndex = Math.Clamp(p.ClassIndex, 0, _labels.Length - 1);
            return new LandmarkPrediction
            {
                Label = _labels[safeIndex],
                Confidence = p.Confidence,
                Rank = idx + 1
            };
        }).ToList();

        return new PredictionResult
        {
            Predictions = mapped,
            InferenceTimeMs = 0 // set by caller
        };
    }

    private async Task CreatePythonScript(string scriptPath)
{
    var pythonScript = @"
import sys
import json
import numpy as np
import tensorflow as tf
from PIL import Image

# version: 2026-01-03_backend_input_fix_v3

def predict(model_path, image_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    ide = interpreter.get_input_details()[0]
    ode = interpreter.get_output_details()[0]
    expected_classes = int(sys.argv[3]) if len(sys.argv) > 3 else None

    num_classes = int(ode['shape'][-1])
    if expected_classes is not None and num_classes != expected_classes:
        sys.stderr.write(
            f'Label/model mismatch: model outputs {num_classes} classes but labels length is {expected_classes}\n'
        )
        sys.exit(1)

    img = Image.open(image_path).convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)

    # IMPORTANT: keep 0..255 float32 (model has preprocess inside)
    x = np.array(img, dtype=np.float32)[None, ...]

    # ---- SAFE DEBUG (no nested quotes) ----
    try:
        in_dtype = ide['dtype']
        in_quant = ide.get('quantization')
        x_min = float(x.min())
        x_max = float(x.max())
        sys.stderr.write(
            f'[debug] in_dtype={in_dtype} in_quant={in_quant} x_min={x_min:.1f} x_max={x_max:.1f}\n'
        )
    except Exception:
        pass

    # Quantize input if needed
    if ide['dtype'] in (np.uint8, np.int8):
        scale, zp = ide.get('quantization', (0.0, 0))
        if scale and scale != 0:
            x = np.round(x / scale + zp).astype(ide['dtype'])
        else:
            x = x.astype(ide['dtype'])

    interpreter.set_tensor(ide['index'], x)
    interpreter.invoke()

    out = interpreter.get_tensor(ode['index'])[0]

    # Dequantize output if needed
    if ode['dtype'] in (np.uint8, np.int8):
        scale, zp = ode.get('quantization', (0.0, 0))
        if scale and scale != 0:
            out = (out.astype(np.float32) - zp) * scale

    top3_idx = np.argsort(out)[-3:][::-1]
    result = [{'class_index': int(idx), 'confidence': float(out[idx])} for idx in top3_idx]
    print(json.dumps(result))

if __name__ == '__main__':
    predict(sys.argv[1], sys.argv[2])
";
    await File.WriteAllTextAsync(scriptPath, pythonScript);
    _logger.LogInformation("Created Python inference script at {ScriptPath}", scriptPath);
}

    public void Dispose()
    {
        // nothing to dispose
    }

    private sealed class PythonPrediction
    {
        [JsonPropertyName("class_index")]
        public int ClassIndex { get; set; }

        [JsonPropertyName("confidence")]
        public float Confidence { get; set; }
    }
}