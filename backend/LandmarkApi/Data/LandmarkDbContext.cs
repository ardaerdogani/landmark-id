using Microsoft.EntityFrameworkCore;
using LandmarkApi.Models;

namespace LandmarkApi.Data;

public class LandmarkDbContext : DbContext
{
    public LandmarkDbContext(DbContextOptions<LandmarkDbContext> options) : base(options)
    {
    }

    public DbSet<PredictionRecord> PredictionRecords { get; set; } = null!;
    public DbSet<PredictionDetail> PredictionDetails { get; set; } = null!;
    public DbSet<PredictionFeedbackRecord> PredictionFeedbackRecords => Set<PredictionFeedbackRecord>();

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        base.OnModelCreating(modelBuilder);

        modelBuilder.Entity<PredictionRecord>()
            .HasMany(r => r.Details)
            .WithOne(d => d.PredictionRecord)
            .HasForeignKey(d => d.PredictionRecordId)
            .OnDelete(DeleteBehavior.Cascade);
    }
}