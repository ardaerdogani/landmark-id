using Microsoft.AspNetCore.Mvc;

namespace LandmarkApi.Controllers;

[ApiController]
[Route("api/[controller]")]
public class DatasetController : ControllerBase
{
    private readonly IConfiguration _config;
    private readonly IWebHostEnvironment _env;
    private readonly ILogger<DatasetController> _logger;

    private static readonly HashSet<string> AllowedExt = new(StringComparer.OrdinalIgnoreCase)
    {
        ".jpg", ".jpeg", ".png"
    };

    public DatasetController(IConfiguration config, IWebHostEnvironment env, ILogger<DatasetController> logger)
    {
        _config = config;
        _env = env;
        _logger = logger;
    }

    [HttpPost("add")]
    [RequestSizeLimit(15 * 1024 * 1024)]
    public async Task<IActionResult> Add([FromForm] string label, [FromForm] IFormFile imageFile)
    {
        if (string.IsNullOrWhiteSpace(label))
            return BadRequest(new { error = "Missing label" });

        if (imageFile is null || imageFile.Length == 0)
            return BadRequest(new { error = "Missing imageFile" });

        var ext = Path.GetExtension(imageFile.FileName);
        if (string.IsNullOrWhiteSpace(ext) || !AllowedExt.Contains(ext))
            return BadRequest(new { error = "Invalid file extension", ext });

        // Validate label exists in backend labels file
        var labelsPath = Path.Combine(_env.ContentRootPath, "Models", "labels.txt");
        if (!System.IO.File.Exists(labelsPath))
            return StatusCode(500, new { error = "labels.txt not found", labelsPath });

        var allowedLabels = System.IO.File.ReadAllLines(labelsPath)
            .Select(x => x.Trim())
            .Where(x => !string.IsNullOrWhiteSpace(x))
            .ToHashSet(StringComparer.OrdinalIgnoreCase);

        if (!allowedLabels.Contains(label))
            return BadRequest(new { error = "Unknown label", label });

        var datasetRootSetting = _config["DatasetRootPath"];
        if (string.IsNullOrWhiteSpace(datasetRootSetting))
            return StatusCode(500, new { error = "DatasetRootPath is not configured" });

        var datasetRoot = Path.IsPathRooted(datasetRootSetting)
            ? datasetRootSetting
            : Path.GetFullPath(Path.Combine(_env.ContentRootPath, datasetRootSetting));

        var targetDir = Path.Combine(datasetRoot, label);
        Directory.CreateDirectory(targetDir);

        var fileName = $"_added_by_user_{DateTime.UtcNow:yyyyMMdd_HHmmss}_{Guid.NewGuid():N}{ext}";
        var fullPath = Path.Combine(targetDir, fileName);

        await using (var fs = System.IO.File.Create(fullPath))
            await imageFile.CopyToAsync(fs);

        _logger.LogInformation("Dataset add: saved {Label} -> {Path}", label, fullPath);

        return Ok(new { label, fileName });
    }
}