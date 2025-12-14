using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Http;
using System.Threading.Tasks;
using System.Text.Json;
using Microsoft.Extensions.Logging;

namespace LandmarkApi.Services;

public class WikipediaService
{
    private readonly HttpClient _httpClient;
    private readonly ILogger<WikipediaService> _logger;
    private const string WikipediaApiUrl = "https://en.wikipedia.org/w/api.php";

    private static readonly Dictionary<string, string> LandmarkNameMapping = new()
    {
        { "Parnid%C5%BEio_kopa", "Parnidis Dune" },
        { "U%C5%BEupis", "Uzupis" },
        { "Freedom_Avenue_(Kaunas)", "Freedom Avenue" },
        { "Trakai_Island_Castle", "Trakai Island Castle" },
        { "Hill_of_Crosses", "Hill of Crosses" },
        { "Rasos_Cemetery", "Rasos Cemetery" },
        { "IX_Fort", "Ninth Fort" },
        { "Gediminas_Tower", "Gediminas Tower" },
        { "Gate_of_Dawn", "Gate of Dawn" },
        { "Zarasas", "Zarasai" }
    };

    public WikipediaService(HttpClient httpClient, ILogger<WikipediaService> logger)
    {
        _httpClient = httpClient;
        _logger = logger;
        
        // Set User-Agent to avoid 403 Forbidden
        if (!_httpClient.DefaultRequestHeaders.Contains("User-Agent"))
        {
            _httpClient.DefaultRequestHeaders.Add("User-Agent", "LandmarkIdentifierApp/1.0 (Educational Project)");
        }
    }

    public async Task<string?> GetWikipediaUrlAsync(string landmarkName)
    {
        try
        {
            _logger.LogInformation($"[Wikipedia] Processing landmark: {landmarkName}");
            
            string searchTerm = LandmarkNameMapping.ContainsKey(landmarkName) 
                ? LandmarkNameMapping[landmarkName]
                : WebUtility.UrlDecode(landmarkName).Replace("_", " ");
            
            _logger.LogInformation($"[Wikipedia] Mapped search term: {searchTerm}");
            
            var url = $"{WikipediaApiUrl}?action=query&format=json&titles={Uri.EscapeDataString(searchTerm)}&prop=info&inprop=url&redirects=1";

            var response = await _httpClient.GetAsync(url);
            response.EnsureSuccessStatusCode();

            var jsonContent = await response.Content.ReadAsStringAsync();
            
            using (JsonDocument doc = JsonDocument.Parse(jsonContent))
            {
                var root = doc.RootElement;
                var query = root.GetProperty("query");
                var pages = query.GetProperty("pages");
                
                foreach (var page in pages.EnumerateObject())
                {
                    if (page.Value.TryGetProperty("fullurl", out var urlElement))
                    {
                        var wikiUrl = urlElement.GetString();
                        _logger.LogInformation($"[Wikipedia] SUCCESS - Found URL: {wikiUrl}");
                        return wikiUrl;
                    }
                }
            }

            _logger.LogWarning($"[Wikipedia] No URL found for: {searchTerm}");
            return null;
        }
        catch (Exception ex)
        {
            _logger.LogError($"[Wikipedia] ERROR for {landmarkName}: {ex.Message}");
            return null;
        }
    }
}