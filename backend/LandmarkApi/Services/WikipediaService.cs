using System.Net;
using System.Text.Json;

namespace LandmarkApi.Services;

public class WikipediaService
{
    private readonly HttpClient _http;
    private readonly ILogger<WikipediaService> _logger;

    private readonly Dictionary<string, string?> _cache = new(StringComparer.OrdinalIgnoreCase);

    private const string KtuWikipediaUrl = "https://en.wikipedia.org/wiki/Kaunas_University_of_Technology";

    // Hard overrides when you know the correct EN article
    private static readonly Dictionary<string, string> LabelToWikipediaUrl = new(StringComparer.OrdinalIgnoreCase)
    {
        { "Istorine_prezidentura", "https://en.wikipedia.org/wiki/Historical_Presidential_Palace,_Kaunas" },
        { "Sv._Arkangelo_Mykolo_baznycia", "https://lt.wikipedia.org/wiki/%C5%A0v._arkangelo_Mykolo_ba%C5%BEny%C4%8Dia" },

    };

    private static readonly Dictionary<string, string> Alias = new(StringComparer.OrdinalIgnoreCase)
    {
        { "Gedimino_pilis", "Gedimino pilis" },
        { "Istorine_prezidentura", "Istorinė Prezidentūra, Kaunas" },
        { "Kauno_pilis", "Kauno pilis" },
        { "Kauno_rotuse", "Kauno rotušė" },
        { "MO_muziejus", "MO muziejus" },
        { "Vilniaus_arkikatedra_bazilika", "Vilniaus arkikatedra bazilika" },
        { "Vytauto_Didziojo_baznycia", "Vytauto Didžiojo bažnyčia" },
        { "Sv._Arkangelo_Mykolo_baznycia", "Šv. Arkangelo Mykolo bažnyčia (Kaunas)" },
        { "Sv._apastalu_Petro_ir_Povilo_arkikatedra_bazilika", "Šv. apaštalų Petro ir Povilo arkikatedra bazilika" },
        { "Kauno_valstybinis_muzikinis_teatras", "Kauno valstybinis muzikinis teatras" },
    };

    public WikipediaService(HttpClient http, ILogger<WikipediaService> logger)
    {
        _http = http;
        _logger = logger;
        _http.Timeout = TimeSpan.FromSeconds(10);

        if (!_http.DefaultRequestHeaders.Contains("User-Agent"))
            _http.DefaultRequestHeaders.Add("User-Agent", "LandmarkIdentifierApp/1.0 (Educational Project)");
    }

    public async Task<string?> GetWikipediaUrlAsync(string label)
    {
        if (string.IsNullOrWhiteSpace(label))
            return null;

        if (_cache.TryGetValue(label, out var cached))
            return cached;

        // 0) Direct URL overrides (preferred)
        if (LabelToWikipediaUrl.TryGetValue(label, out var forcedUrl))
            return _cache[label] = forcedUrl;

        // 1) KTU shortcut
        if (label.StartsWith("KTU_", StringComparison.OrdinalIgnoreCase))
            return _cache[label] = KtuWikipediaUrl;

        var query = Alias.TryGetValue(label, out var alias)
            ? alias
            : WebUtility.UrlDecode(label).Replace('_', ' ').Trim();

        var langs = new[] { "lt", "en" };

        foreach (var lang in langs)
        {
            var exactTitle = await TryGetExistingTitleAsync(lang, query);
            if (!string.IsNullOrWhiteSpace(exactTitle))
                return _cache[label] = MakeWikiUrl(lang, exactTitle!);

            var searchTitle = await TryOpenSearchTopTitleAsync(lang, query);
            if (!string.IsNullOrWhiteSpace(searchTitle))
                return _cache[label] = MakeWikiUrl(lang, searchTitle!);
        }

        _cache[label] = null;
        return null;
    }

    private static string MakeWikiUrl(string lang, string title)
    {
        var slug = title.Replace(' ', '_');
        return $"https://{lang}.wikipedia.org/wiki/{Uri.EscapeDataString(slug)}";
    }

    private async Task<string?> TryGetExistingTitleAsync(string lang, string title)
    {
        var url =
            $"https://{lang}.wikipedia.org/w/api.php" +
            $"?action=query&format=json&origin=*&redirects=1&titles={Uri.EscapeDataString(title)}";

        try
        {
            using var resp = await _http.GetAsync(url);
            if (!resp.IsSuccessStatusCode) return null;

            var json = await resp.Content.ReadAsStringAsync();
            using var doc = JsonDocument.Parse(json);

            if (!doc.RootElement.TryGetProperty("query", out var queryEl)) return null;
            if (!queryEl.TryGetProperty("pages", out var pagesEl)) return null;

            foreach (var pageProp in pagesEl.EnumerateObject())
            {
                var page = pageProp.Value;
                if (page.TryGetProperty("missing", out _)) return null;

                if (page.TryGetProperty("title", out var titleEl))
                    return titleEl.GetString();
            }

            return null;
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "Wikipedia exact-title lookup failed for {Lang}:{Title}", lang, title);
            return null;
        }
    }

    private async Task<string?> TryOpenSearchTopTitleAsync(string lang, string search)
    {
        var url =
            $"https://{lang}.wikipedia.org/w/api.php" +
            $"?action=opensearch&format=json&origin=*&limit=1&namespace=0&search={Uri.EscapeDataString(search)}";

        try
        {
            using var resp = await _http.GetAsync(url);
            if (!resp.IsSuccessStatusCode) return null;

            var json = await resp.Content.ReadAsStringAsync();
            using var doc = JsonDocument.Parse(json);

            if (doc.RootElement.ValueKind != JsonValueKind.Array) return null;
            if (doc.RootElement.GetArrayLength() < 2) return null;

            var titlesArr = doc.RootElement[1];
            if (titlesArr.ValueKind != JsonValueKind.Array) return null;
            if (titlesArr.GetArrayLength() < 1) return null;

            return titlesArr[0].GetString();
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "Wikipedia search lookup failed for {Lang}:{Search}", lang, search);
            return null;
        }
    }
}