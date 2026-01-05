using LandmarkApi.Data;
using LandmarkApi.Models;
using Microsoft.AspNetCore.Mvc;

namespace LandmarkApi.Controllers;

[ApiController]
[Route("api/[controller]")]
public class FeedbackController : ControllerBase
{
    private readonly LandmarkDbContext _db;
    private readonly ILogger<FeedbackController> _logger;

    public FeedbackController(LandmarkDbContext db, ILogger<FeedbackController> logger)
    {
        _db = db;
        _logger = logger;
    }

    public sealed class CreateFeedbackRequest
    {
        public int? PredictionRecordId { get; set; }
        public string? PredictedLabel { get; set; }
        public float? PredictedConfidence { get; set; }
        public string? CorrectLabel { get; set; }
        public string? Comment { get; set; }
    }

    [HttpPost]
    public async Task<IActionResult> Create([FromBody] CreateFeedbackRequest req)
    {
        if (req is null)
            return BadRequest(new { error = "Missing request body" });

        var feedback = new PredictionFeedbackRecord
        {
            PredictionRecordId = req.PredictionRecordId,
            PredictedLabel = req.PredictedLabel,
            PredictedConfidence = req.PredictedConfidence,
            CorrectLabel = req.CorrectLabel,
            Comment = req.Comment,
            CreatedAtUtc = DateTime.UtcNow
        };

        _db.PredictionFeedbackRecords.Add(feedback);
        await _db.SaveChangesAsync();

        _logger.LogInformation("Saved feedback {Id} for prediction {PredictionRecordId}", feedback.Id, feedback.PredictionRecordId);

        return Ok(new { id = feedback.Id });
    }
}