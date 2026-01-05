using System.ComponentModel.DataAnnotations;

namespace LandmarkApi.Models;

public class PredictionFeedbackRecord
{
    public int Id { get; set; }

    // Link to a saved prediction (optional but useful)
    public int? PredictionRecordId { get; set; }
    public PredictionRecord? PredictionRecord { get; set; }

    [MaxLength(200)]
    public string? PredictedLabel { get; set; }

    public float? PredictedConfidence { get; set; }

    [MaxLength(200)]
    public string? CorrectLabel { get; set; }

    [MaxLength(2000)]
    public string? Comment { get; set; }

    public DateTime CreatedAtUtc { get; set; } = DateTime.UtcNow;
}