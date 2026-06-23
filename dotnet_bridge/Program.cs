using Microsoft.AspNetCore.Mvc;
using System.Text.Json;
using System.IO;
using System.Net.Http;
using System.Text;
using System.Text.RegularExpressions;
using StackExchange.Redis;
using MySqlConnector;

var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();

// Initialize Database on startup
InitializeDatabase();

// Start the Doctor background self-learning and log watcher thread
_ = Task.Run(async () => {
    await Task.Delay(10000); // Wait for the main container to settle
    await StartDoctorEngineAsync();
});

app.MapPost("/process", ([FromBody] RequestContext context) => {
    // 1. Block admin requests without appropriate key
    if (context.Path.StartsWith("/admin", StringComparison.OrdinalIgnoreCase) && 
        (!context.Headers.TryGetValue("x-admin-token", out var adminToken) || adminToken != "secret_admin_token")) 
    {
        return Results.Json(new MiddlewareResponse {
            Action = "block",
            StatusCode = 403,
            Detail = "Blocked by .NET Core Bridge: Unauthorized admin path access."
        });
    }

    // 2. Reject extremely long requests to prevent DoS
    if (context.Body != null && context.Body.Length > 100000)
    {
        return Results.Json(new MiddlewareResponse {
            Action = "block",
            StatusCode = 413,
            Detail = "Blocked by .NET Core Bridge: Request body payload is too large."
        });
    }

    // 3. Inject tracing headers for audit
    var modifiedHeaders = new Dictionary<string, string>(context.Headers, StringComparer.OrdinalIgnoreCase);
    modifiedHeaders["X-Processed-By-DotNet-Bridge"] = "true";

    return Results.Json(new MiddlewareResponse {
        Action = "allow",
        ModifiedHeaders = modifiedHeaders
    });
});

app.MapPost("/apply-upgrade", ([FromBody] UpgradeRequest request) => {
    string rootDir = Directory.Exists("/app_host") ? "/app_host" : ".";
    string backupsDir = Path.Combine(rootDir, "backups");
    
    var result = ApplyUpgradeInternal(rootDir, request, backupsDir);
    return Results.Json(result);
});

app.MapPost("/analyze-and-upgrade", async ([FromBody] AnalyzeRequest request) => {
    string rootDir = Directory.Exists("/app_host") ? "/app_host" : ".";
    string targetFile = Path.Combine(rootDir, request.FilePath ?? "app/main.py");
    string backupsDir = Path.Combine(rootDir, "backups");

    if (!File.Exists(targetFile)) {
        return Results.Json(new UpgradeResponse { Success = false, Error = $"Target file not found: {request.FilePath}" });
    }

    Console.WriteLine($"[Doctor] Manual optimization requested for: {targetFile}");
    try {
        bool success = await RunDarwinianMutationCycleAsync(rootDir, backupsDir, targetFile, request.CustomPrompt ?? "Optimize performance, reduce latency, and ensure strict PEP8 guidelines.");
        return Results.Json(new UpgradeResponse { Success = success, Error = success ? "" : "Optimization failed or shadow sandbox rejected mutation." });
    } catch (Exception ex) {
        return Results.Json(new UpgradeResponse { Success = false, Error = ex.Message });
    }
});

app.Run("http://127.0.0.1:5000");

// Parse MySQL Connection String from DATABASE_URL
string GetMySqlConnectionString()
{
    var dbUrl = Environment.GetEnvironmentVariable("DATABASE_URL") ?? "mysql+pymysql://root:@host.docker.internal:3306/agent_db";
    var match = Regex.Match(dbUrl, @"mysql\+pymysql://([^:]*):([^@]*)@([^:/]+)(?::(\d+))?/([^?]+)");
    if (match.Success)
    {
        var uid = match.Groups[1].Value;
        var pwd = match.Groups[2].Value;
        var host = match.Groups[3].Value;
        var port = match.Groups[4].Success ? match.Groups[4].Value : "3306";
        var db = match.Groups[5].Value;
        
        return $"Server={host};Port={port};Database={db};Uid={uid};Pwd={pwd};AllowZeroDateTime=True;ConvertZeroDateTime=True;";
    }
    return "Server=host.docker.internal;Port=3306;Database=agent_db;Uid=root;Pwd=;AllowZeroDateTime=True;ConvertZeroDateTime=True;";
}

// Parse Redis Configuration from REDIS_URL
string GetRedisConfig()
{
    var redisUrl = Environment.GetEnvironmentVariable("REDIS_URL") ?? "redis://redis:6379/0";
    var match = Regex.Match(redisUrl, @"redis://([^:/]+)(?::(\d+))?");
    if (match.Success)
    {
        var host = match.Groups[1].Value;
        var port = match.Groups[2].Success ? match.Groups[2].Value : "6379";
        return $"{host}:{port}";
    }
    return "redis:6379";
}

// Initialize MySQL Tables
void InitializeDatabase()
{
    try
    {
        var connStr = GetMySqlConnectionString();
        using var conn = new MySqlConnection(connStr);
        conn.Open();
        
        using var cmd = conn.CreateCommand();
        cmd.CommandText = @"
            CREATE TABLE IF NOT EXISTS code_backups (
                id VARCHAR(50) PRIMARY KEY,
                file_path VARCHAR(255) NOT NULL,
                backup_file_name VARCHAR(255) NOT NULL,
                timestamp DATETIME NOT NULL,
                status VARCHAR(20) NOT NULL
            );
            CREATE TABLE IF NOT EXISTS evolution_logs (
                id VARCHAR(50) PRIMARY KEY,
                file_path VARCHAR(255) NOT NULL,
                from_version VARCHAR(50) NOT NULL,
                to_version VARCHAR(50) NOT NULL,
                mutated_code TEXT,
                latency_change_ms INT,
                ram_usage_bytes BIGINT,
                status VARCHAR(20) NOT NULL,
                failure_reason TEXT,
                timestamp DATETIME NOT NULL
            );
            CREATE TABLE IF NOT EXISTS checkpoints (
                id VARCHAR(50) PRIMARY KEY,
                task_description VARCHAR(255) NOT NULL,
                status VARCHAR(20) NOT NULL,
                timestamp DATETIME NOT NULL
            );
            CREATE TABLE IF NOT EXISTS punishment_logs (
                id VARCHAR(50) PRIMARY KEY,
                cycle_number INT NOT NULL DEFAULT 0,
                ram_baseline_mb FLOAT NOT NULL DEFAULT 0,
                ram_mutant_mb FLOAT NOT NULL DEFAULT 0,
                ram_excess_mb FLOAT NOT NULL DEFAULT 0,
                storage_free_gb FLOAT NOT NULL DEFAULT 0,
                punishment_message TEXT NOT NULL,
                timestamp DATETIME NOT NULL
            );
            CREATE TABLE IF NOT EXISTS doctor_lifecycle_logs (
                id VARCHAR(50) PRIMARY KEY,
                cycle_id VARCHAR(50) NOT NULL,
                event_type VARCHAR(50) NOT NULL,
                details TEXT NOT NULL,
                timestamp DATETIME NOT NULL
            );"; 
        cmd.ExecuteNonQuery();
        Console.WriteLine("[Doctor DB] MySQL Database tables checked/initialized successfully.");
    }
    catch (Exception ex)
    {
        Console.WriteLine($"[Doctor DB Error] Failed to initialize database: {ex.Message}");
    }
}

// Internal helper to apply upgrade and backup
UpgradeResponse ApplyUpgradeInternal(string rootDir, UpgradeRequest request, string backupsDir) {
    try {
        string targetFilePath = Path.GetFullPath(Path.Combine(rootDir, request.FilePath));
        string resolvedRoot = Path.GetFullPath(rootDir);
        
        if (!targetFilePath.StartsWith(resolvedRoot)) {
            return new UpgradeResponse { Success = false, Error = "Security Exception: Path traversal attempt blocked." };
        }
        
        if (File.Exists(targetFilePath)) {
            string currentContent = File.ReadAllText(targetFilePath);
            SaveBackupToDB(request.FilePath, targetFilePath, currentContent, backupsDir);
        }

        if (request.Action == "write") {
            Directory.CreateDirectory(Path.GetDirectoryName(targetFilePath)!);
            File.WriteAllText(targetFilePath, request.Content);
            Console.WriteLine($"[Doctor] Wrote new content to {request.FilePath}");
        } else if (request.Action == "patch") {
            if (File.Exists(targetFilePath)) {
                string currentContent = File.ReadAllText(targetFilePath);
                if (currentContent.Contains(request.SearchContent)) {
                    string updatedContent = currentContent.Replace(request.SearchContent, request.Content);
                    File.WriteAllText(targetFilePath, updatedContent);
                    Console.WriteLine($"[Doctor] Patched {request.FilePath}");
                } else {
                    return new UpgradeResponse { Success = false, Error = "Search content not found for patching." };
                }
            } else {
                return new UpgradeResponse { Success = false, Error = "File not found for patching." };
            }
        }
        
        return new UpgradeResponse { Success = true };
    } catch (Exception ex) {
        return new UpgradeResponse { Success = false, Error = ex.Message };
    }
}

// Watch log file and trigger learning when idle
async Task StartDoctorEngineAsync()
{
    string rootDir = Directory.Exists("/app_host") ? "/app_host" : ".";
    string logPath = Path.Combine(rootDir, "fastapi.log");
    string backupsDir = Path.Combine(rootDir, "backups");
    string redisConfig = GetRedisConfig();
    
    ConnectionMultiplexer? redis = null;
    IDatabase? redisDb = null;
    
    try
    {
        redis = ConnectionMultiplexer.Connect(redisConfig);
        redisDb = redis.GetDatabase();
        Console.WriteLine($"[Doctor] Connected to Redis successfully: {redisConfig}");
    }
    catch (Exception rx)
    {
        Console.WriteLine($"[Doctor WARNING] Could not connect to Redis: {rx.Message}. Falling back to time-based idle checking.");
    }

    Console.WriteLine($"[Doctor Engine] Log watcher started. Watching log path: {logPath}");
    long lastPosition = 0;
    if (File.Exists(logPath))
    {
        lastPosition = new FileInfo(logPath).Length;
    }
    
    DateTime lastUserActivity = DateTime.UtcNow;
    var recentErrors = new List<string>();         // Buffer for grouping burst errors
    DateTime lastErrorSeen = DateTime.MinValue;    // Timestamp of last detected error
    const int ERROR_FLUSH_SECONDS = 8;             // Collect errors for 8s before triggering fix
    const int ERROR_COOLDOWN_SECONDS = 60;         // Min gap between successive error fixes
    
    while (true)
    {
        await Task.Delay(5000);
        
        try
        {
            // 1. Process ERROR lines from FastAPI log (Reactive Physician — highest priority)
            if (File.Exists(logPath))
            {
                using (var fs = new FileStream(logPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
                {
                    if (fs.Length < lastPosition) lastPosition = 0;
                    if (fs.Length > lastPosition)
                    {
                        fs.Seek(lastPosition, SeekOrigin.Begin);
                        using (var reader = new StreamReader(fs))
                        {
                            string newContent = await reader.ReadToEndAsync();
                            lastPosition = fs.Position;
                            
                            var lines = newContent.Split(new[] { "\r\n", "\r", "\n" }, StringSplitOptions.RemoveEmptyEntries);
                            foreach (var line in lines)
                            {
                                // Catch ALL FastAPI ERROR-level log lines (standard Python logging format)
                                bool isError = line.Contains(" - ERROR - ") ||
                                               line.Contains("EXCEPTION_ERROR") ||
                                               line.Contains("Traceback (most recent call last):") ||
                                               line.Contains("Exception:") ||
                                               line.Contains("Error:") && line.Contains("agent.");

                                if (isError)
                                {
                                    Console.WriteLine($"[Doctor] 🚨 ERROR detected in FastAPI log: {line.Trim()}");
                                    recentErrors.Add(line.Trim());
                                    lastErrorSeen = DateTime.UtcNow;
                                }
                            }
                        }
                    }
                }
            }

            // 2. Flush accumulated errors after the burst window — then trigger IMMEDIATE repair
            bool hasPendingErrors = recentErrors.Count > 0 &&
                                    (DateTime.UtcNow - lastErrorSeen).TotalSeconds >= ERROR_FLUSH_SECONDS;
            bool isCooldownExpired = (DateTime.UtcNow - lastErrorSeen).TotalSeconds < (ERROR_FLUSH_SECONDS + ERROR_COOLDOWN_SECONDS) == false;

            if (hasPendingErrors)
            {
                var errorBlock = string.Join("\n", recentErrors.TakeLast(15)); // cap at 15 lines
                Console.WriteLine($"[Doctor] 🚨 Flushing {recentErrors.Count} error(s). Triggering IMMEDIATE repair cycle...");
                recentErrors.Clear();

                // Override any LEARNING status — error repair is URGENT
                if (redisDb != null)
                {
                    var currentStatus = await redisDb.StringGetAsync("system_status");
                    if (currentStatus == "LEARNING")
                    {
                        Console.WriteLine("[Doctor] Preempting LEARNING cycle with URGENT error repair.");
                        await redisDb.StringSetAsync("system_status", "ERROR_REPAIR", TimeSpan.FromMinutes(10));
                    }
                    else if (currentStatus == "USER_PRIORITY" || currentStatus == "BUSY")
                    {
                        // Queue the error fix for after user is done — don't interrupt
                        Console.WriteLine($"[Doctor] User is active ({currentStatus}). Queuing error repair after user session.");
                        await redisDb.StringSetAsync("queued_error_repair", errorBlock, TimeSpan.FromMinutes(15));
                        goto SkipIdleCheck;
                    }
                    else
                    {
                        await redisDb.StringSetAsync("system_status", "ERROR_REPAIR", TimeSpan.FromMinutes(10));
                    }
                }

                string errorCycleId = Guid.NewGuid().ToString();
                SaveLifecycleLog(errorCycleId, "ErrorRepairTriggered", $"Urgent repair triggered due to FastAPI log errors:\n{errorBlock}");
                string repairInstruction = BuildErrorRepairPrompt(errorBlock);
                bool repaired = await RunDarwinianMutationCycleAsync(rootDir, backupsDir, Path.Combine(rootDir, "app/main.py"), repairInstruction, redisDb, errorCycleId);

                if (redisDb != null)
                    await redisDb.KeyDeleteAsync("system_status"); // Release lock after repair

                Console.WriteLine(repaired
                    ? "[Doctor] ✅ Repair cycle completed. FastAPI should auto-reload with fix."
                    : "[Doctor] ⚠️ Repair cycle did not produce a valid fix. Will retry next error burst.");

                goto SkipIdleCheck; // Skip idle check this loop iteration
            }

            // 3. Check for queued error repair (user was busy when error occurred)
            if (redisDb != null)
            {
                var queuedError = await redisDb.StringGetAsync("queued_error_repair");
                var currentRedisStatus = await redisDb.StringGetAsync("system_status");
                if (!queuedError.IsNullOrEmpty && currentRedisStatus.IsNullOrEmpty)
                {
                    Console.WriteLine("[Doctor] 📥 Found queued error repair from user session. Executing now...");
                    await redisDb.KeyDeleteAsync("queued_error_repair");
                    await redisDb.StringSetAsync("system_status", "ERROR_REPAIR", TimeSpan.FromMinutes(10));
                    
                    string queuedCycleId = Guid.NewGuid().ToString();
                    SaveLifecycleLog(queuedCycleId, "ErrorRepairTriggered", $"Queued error repair triggered from previous user session errors:\n{queuedError}");
                    string queuedInstruction = BuildErrorRepairPrompt(queuedError.ToString());
                    await RunDarwinianMutationCycleAsync(rootDir, backupsDir, Path.Combine(rootDir, "app/main.py"), queuedInstruction, redisDb, queuedCycleId);
                    await redisDb.KeyDeleteAsync("system_status");
                    goto SkipIdleCheck;
                }
            }

            // 4. Heartbeat Idle and Pause-Continue Check
            if (redisDb != null)
            {
                var status = await redisDb.StringGetAsync("system_status");
                
                if (status.IsNullOrEmpty)
                {
                    // Redis system_status key has expired, meaning system has been idle for 5+ minutes
                    Console.WriteLine("[Doctor Engine] System is IDLE (5+ minutes of inactivity). Awakening Self-Learning process...");
                    
                    // Show real-time resource snapshot before starting mutation
                    var idleSnapshot = GetResourceSnapshot();
                    Console.WriteLine($"[Doctor Resource] 🧠 RAM: {idleSnapshot.RamUsedMb:F0}MB / {(idleSnapshot.RamLimitMb > 0 ? idleSnapshot.RamLimitMb.ToString("F0") + "MB" : "?")} ({idleSnapshot.RamUsagePct:F0}%) — Status: {idleSnapshot.RamStatus}");
                    Console.WriteLine($"[Doctor Resource] 💾 Storage: {idleSnapshot.StorageFreeGb:F1}GB free / {idleSnapshot.StorageTotalGb:F1}GB — Status: {idleSnapshot.StorageStatus}");
                    
                    // Lock learning status
                    await redisDb.StringSetAsync("system_status", "LEARNING");
                    
                    // Check if we have a paused checkpoint to continue
                    var checkpoint = GetLatestCheckpoint();
                    string baseInstruction = "Optimize performance, reduce latency, and ensure strict PEP8 guidelines.";
                    string selfLearningCycleId = Guid.NewGuid().ToString();
                    
                    if (checkpoint != null)
                    {
                        SaveLifecycleLog(selfLearningCycleId, "CycleContinued", $"Resuming paused self-learning checkpoint. Last task: {checkpoint.TaskDescription}");
                        Console.WriteLine($"[Doctor Checkpoint] Resuming paused self-learning checkpoint: {checkpoint.TaskDescription}");
                        baseInstruction = $"Continue our optimization. Last task details: {checkpoint.TaskDescription}";
                    }
                    else
                    {
                        SaveLifecycleLog(selfLearningCycleId, "CycleStart", "Triggering new idle self-learning cycle.");
                    }
                    
                    bool mutationFinished = await RunDarwinianMutationCycleAsync(rootDir, backupsDir, Path.Combine(rootDir, "app/main.py"), baseInstruction, redisDb, selfLearningCycleId);
                    
                    if (mutationFinished)
                    {
                        // Set status back to idle/none to allow the next cycle
                        await redisDb.KeyDeleteAsync("system_status");
                        ClearCheckpoints();
                    }
                }
                else if (status == "BUSY" || status == "USER_PRIORITY")
                {
                    // User is currently active, update last activity time
                    lastUserActivity = DateTime.UtcNow;
                }
            }
            else
            {
                // Fallback time-based idle checking (if Redis connection is down)
                if ((DateTime.UtcNow - lastUserActivity).TotalMinutes >= 5)
                {
                    Console.WriteLine("[Doctor Engine] System is IDLE (5+ minutes). Triggering fallback self-learning cycle...");
                    var fallbackSnapshot = GetResourceSnapshot();
                    Console.WriteLine($"[Doctor Resource] 🧠 RAM: {fallbackSnapshot.RamUsedMb:F0}MB ({fallbackSnapshot.RamStatus}) | 💾 Storage: {fallbackSnapshot.StorageFreeGb:F1}GB free ({fallbackSnapshot.StorageStatus})");
                    
                    string fallbackCycleId = Guid.NewGuid().ToString();
                    SaveLifecycleLog(fallbackCycleId, "CycleStart", "Triggering fallback self-learning cycle (Redis down).");
                    await RunDarwinianMutationCycleAsync(rootDir, backupsDir, Path.Combine(rootDir, "app/main.py"), "Optimize app/main.py performance and reduce memory footprint.", null, fallbackCycleId);
                    lastUserActivity = DateTime.UtcNow;
                }
            }

            SkipIdleCheck:;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Doctor Engine Error] Exception in loop: {ex.Message}");
        }
    }
}

// Darwinian Mutation Cycle: Optimize, Sandbox, Survival Test, deploy or punish
async Task<bool> RunDarwinianMutationCycleAsync(string rootDir, string backupsDir, string targetFilePath, string promptContext, IDatabase? redisDb = null, string? cycleId = null)
{
    string activeCycleId = cycleId ?? Guid.NewGuid().ToString();
    string targetRelativePath = Path.GetRelativePath(rootDir, targetFilePath).Replace("\\", "/");
    string mutatedFilePath = Path.Combine(rootDir, "app/main_mutated.py");
    
    // Ensure we start clean
    if (File.Exists(mutatedFilePath)) File.Delete(mutatedFilePath);

    try
    {
        string currentCode = File.ReadAllText(targetFilePath);
        string codeSnippet = currentCode.Length > 12000 ? currentCode.Substring(0, 12000) : currentCode;

        // Resource Constraints monitoring: full RAM + Storage snapshot
        var snapshot = GetResourceSnapshot();
        double baselineRamMb = snapshot.RamUsedMb; // Capture baseline BEFORE launching shadow container
        
        // Log start of cycle
        SaveLifecycleLog(activeCycleId, "CycleStart", $"Starting mutation cycle on file '{targetRelativePath}'. Target file size: {currentCode.Length} chars. Baseline RAM: {snapshot.RamUsedMb:F0}MB/{(snapshot.RamLimitMb > 0 ? snapshot.RamLimitMb.ToString("F0") + "MB" : "unlimited")}. Storage Free: {snapshot.StorageFreeGb:F1}GB.");
        Console.WriteLine($"[Darwinian Engine] Starting mutation cycle on: {targetRelativePath}");
        Console.WriteLine($"[Darwinian Engine] 📊 Resource Snapshot — RAM: {snapshot.RamUsedMb:F0}MB/{(snapshot.RamLimitMb > 0 ? snapshot.RamLimitMb.ToString("F0") : "?")}MB ({snapshot.RamStatus}) | Storage: {snapshot.StorageFreeGb:F1}GB free ({snapshot.StorageStatus})");

        // Retrieve last punishment message to teach AI from past mistakes
        string lastPunishment = GetLastPunishmentMessage();
        if (!string.IsNullOrEmpty(lastPunishment))
            Console.WriteLine("[Darwinian Engine] ⚠️ Previous punishment detected — injecting into survival prompt for learning.");

        string systemPrompt = "You are a professional Python FastAPI architect running inside a resource-constrained Docker environment.\n" +
                              "Return only a valid JSON object matching the mutation schema:\n" +
                              "{\n" +
                              "  \"filePath\": \"" + targetRelativePath + "\",\n" +
                              "  \"action\": \"write\",\n" +
                              "  \"content\": \"<COMPLETE mutated python code for the target file>\",\n" +
                              "  \"triggerRebuild\": false\n" +
                              "}\n" +
                              "Output ONLY the raw JSON block without formatting, explanations, or code fences.";

        string userPrompt = BuildSurvivalPrompt(snapshot, lastPunishment, promptContext, codeSnippet, targetRelativePath);

        var ollamaBaseUrl = Environment.GetEnvironmentVariable("OLLAMA_BASE_URL") ?? "http://ollama:11434";
        var ollamaModel = Environment.GetEnvironmentVariable("OLLAMA_MODEL") ?? "qwen2.5-coder:1.5b";

        using (var cts = new CancellationTokenSource())
        using (var client = new HttpClient())
        {
            client.Timeout = TimeSpan.FromSeconds(600);
            var payload = new
            {
                model = ollamaModel,
                messages = new[]
                {
                    new { role = "system", content = systemPrompt },
                    new { role = "user", content = userPrompt }
                },
                stream = false,
                options = new { temperature = 0.2 }
            };

            var httpContent = new StringContent(JsonSerializer.Serialize(payload), Encoding.UTF8, "application/json");
            
            // Log Ollama Query Start
            SaveLifecycleLog(activeCycleId, "OllamaQueryStart", $"Requesting code improvement/upgrade from Ollama model '{ollamaModel}' with instruction context: '{promptContext}'.");
            
            // Start HttpClient call to Ollama
            var ollamaTask = client.PostAsync($"{ollamaBaseUrl}/api/chat", httpContent, cts.Token);
            
            // Active polling to check if user interrupted the learning cycle
            while (!ollamaTask.IsCompleted)
            {
                await Task.Delay(100);
                if (redisDb != null)
                {
                    var currentStatus = await redisDb.StringGetAsync("system_status");
                    if (currentStatus == "USER_PRIORITY" || currentStatus == "BUSY")
                    {
                        Console.WriteLine("[Darwinian Engine] USER INTERRUPT DETECTED! Aborting Ollama query instantly to free VRAM...");
                        cts.Cancel(); // Force cancel connection
                        
                        // Save checkpoint and log pause
                        SaveLifecycleLog(activeCycleId, "OllamaQueryPaused", $"User interrupt detected! Aborting Ollama query to release VRAM. Saved checkpoint for resume.");
                        SaveCheckpoint(promptContext, "Paused");
                        return false; // Interrupted
                    }
                }
            }

            var response = await ollamaTask;
            if (response.IsSuccessStatusCode)
            {
                var responseContent = await response.Content.ReadAsStringAsync();
                var doc = JsonDocument.Parse(responseContent);
                string rawReply = doc.RootElement.GetProperty("message").GetProperty("content").GetString() ?? "";

                // Clean markdown code blocks from response
                string jsonString = rawReply.Trim();
                if (jsonString.StartsWith("```json")) {
                    jsonString = jsonString.Substring(7);
                } else if (jsonString.StartsWith("```")) {
                    jsonString = jsonString.Substring(3);
                }
                if (jsonString.EndsWith("```")) {
                    jsonString = jsonString.Substring(0, jsonString.Length - 3);
                }
                jsonString = jsonString.Trim();

                UpgradeRequest? mutationReq = null;
                try
                {
                    mutationReq = JsonSerializer.Deserialize<UpgradeRequest>(jsonString, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
                }
                catch (Exception jsonEx)
                {
                    Console.WriteLine($"[Darwinian Engine] Failed to parse Ollama JSON: {jsonEx.Message}. Response was: {rawReply}");
                    SaveLifecycleLog(activeCycleId, "CycleFailed", $"Failed to parse Ollama output JSON: {jsonEx.Message}. Raw reply: {rawReply}");
                    SaveEvolutionLog(targetRelativePath, "unknown", "mutated", "Ollama output json parsing failed", 0, 0, "Gagal", jsonEx.Message);
                    return true;
                }

                if (mutationReq != null && !string.IsNullOrEmpty(mutationReq.Content))
                {
                    // Log Ollama Query Success
                    SaveLifecycleLog(activeCycleId, "OllamaQuerySuccess", $"Ollama returned code mutation response. Code length: {mutationReq.Content.Length} characters.");
                    Console.WriteLine("[Darwinian Sandbox] Mutated code generated. Running quarantine survival testing...");
                    
                    // Write to mutated temporary file
                    File.WriteAllText(mutatedFilePath, mutationReq.Content);
                    
                    // Run survival testing in shadow container sandbox with baseline RAM for Punishment Protocol
                    var survivalResult = await RunQuarantineSandboxTestsAsync(rootDir, mutatedFilePath, baselineRamMb, activeCycleId);
                    
                    if (survivalResult.Lulus)
                    {
                        Console.WriteLine($"[Darwinian Sandbox] SUCCESS! Mutation passed survival tests. Deploying version...");
                        
                        // Save original backup to database
                        string originalContent = File.ReadAllText(targetFilePath);
                        SaveBackupToDB(targetRelativePath, targetFilePath, originalContent, backupsDir);
                        
                        // Deploy: overwrite the main app file
                        File.WriteAllText(targetFilePath, mutationReq.Content);
                        
                        // Log success to evolution logs & lifecycle logs
                        SaveLifecycleLog(activeCycleId, "CycleCompleted", $"Mutation tests passed. Deployed code changes to '{targetRelativePath}' successfully.");
                        SaveEvolutionLog(targetRelativePath, "v_old", "v_new", mutationReq.Content, survivalResult.LatencyDeltaMs, survivalResult.RamUsageBytes, "Lulus", "");
                        
                        // Delete temp file
                        if (File.Exists(mutatedFilePath)) File.Delete(mutatedFilePath);
                        return true;
                    }
                    else
                    {
                        Console.WriteLine($"[Darwinian Sandbox] FAILED: Mutation rejected by sandbox. Reason: {survivalResult.Reason}");
                        
                        // Log failure to lifecycle logs & evolution logs
                        SaveLifecycleLog(activeCycleId, "CycleFailed", $"Mutation rejected by sandbox tests. Reason: {survivalResult.Reason}");
                        SaveEvolutionLog(targetRelativePath, "v_old", "v_new", mutationReq.Content, 0, 0, "Gagal", survivalResult.Reason);
                        
                        if (File.Exists(mutatedFilePath)) File.Delete(mutatedFilePath);
                        return true;
                    }
                }
            }
            else
            {
                SaveLifecycleLog(activeCycleId, "CycleFailed", $"Ollama chat query failed with status code: {response.StatusCode}");
            }
        }
    }
    catch (Exception ex)
    {
        SaveLifecycleLog(activeCycleId, "CycleFailed", $"Mutation cycle encountered an error: {ex.Message}");
        Console.WriteLine($"[Darwinian Engine Error] Mutation cycle exception: {ex.Message}");
    }
    return false;
}

// Spin up shadow container, measure latency/RAM, check validity
async Task<SandboxTestResult> RunQuarantineSandboxTestsAsync(string rootDir, string mutatedFilePath, double baselineRamMb = 0, string cycleId = "unknown")
{
    Console.WriteLine("[Darwinian Sandbox] Launching shadow API container...");
    string containerName = "agent-shadow-api";
    
    // Ensure no existing shadow container is running
    ExecuteTerminalCommand($"docker rm -f {containerName}");

    // Log Sandbox Start
    SaveLifecycleLog(cycleId, "SandboxStart", $"Launching shadow API container '{containerName}' for quarantine verification testing.");

    // Start shadow container in default compose network, mapping the shared volume
    // Uvicorn runs app.main_mutated:app inside the container
    string startCommand = $"docker compose -f {Path.Combine(rootDir, "docker-compose.yml")} run -d --name {containerName} -p 8080:8000 agent-api uvicorn app.main_mutated:app --host 0.0.0.0 --port 8000";
    var startOutput = ExecuteTerminalCommand(startCommand);
    
    if (string.IsNullOrEmpty(startOutput) || startOutput.Contains("Error"))
    {
        SaveLifecycleLog(cycleId, "CompilationFailed", $"Stage 1 failed: Shadow container startup command failed. Output: {startOutput}");
        return new SandboxTestResult(false, $"Shadow container startup command failed: {startOutput}", 0, 0);
    }

    // Log Stage 1: Compilation Check
    SaveLifecycleLog(cycleId, "CompilationCheck", "Stage 1: Verifying compilation and service start health check on shadow container (port 8080).");

    // Wait for the container port 8080 to respond (Uji Kompilasi)
    bool isAlive = false;
    using (var client = new HttpClient())
    {
        client.Timeout = TimeSpan.FromSeconds(1);
        for (int i = 0; i < 15; i++)
        {
            await Task.Delay(1000);
            try
            {
                var res = await client.GetAsync("http://localhost:8080/health");
                if (res.IsSuccessStatusCode)
                {
                    isAlive = true;
                    break;
                }
            }
            catch { }
        }
    }

    if (!isAlive)
    {
        string containerLogs = ExecuteTerminalCommand($"docker logs {containerName}");
        ExecuteTerminalCommand($"docker rm -f {containerName}");
        
        SaveLifecycleLog(cycleId, "CompilationFailed", $"Stage 1 failed. Shadow container did not respond to /health in 15 seconds. Startup logs:\n{containerLogs}");
        return new SandboxTestResult(false, $"Uji Kompilasi GAGAL: Container did not become healthy in 15s. Logs:\n{containerLogs}", 0, 0);
    }

    // Log Stage 1 Success
    SaveLifecycleLog(cycleId, "CompilationSuccess", "Stage 1 passed. Shadow container compiled successfully and is responding on port 8080.");

    // Uji Latensi: Send 100 requests to shadow container and calculate average response time
    int totalRequests = 100;
    long totalMs = 0;
    int successCount = 0;
    
    // Log Stage 2: Stress Test
    SaveLifecycleLog(cycleId, "StressTestStart", $"Stage 2: Initiating stress test and latency check by sending {totalRequests} HTTP requests to shadow container.");

    using (var client = new HttpClient())
    {
        client.Timeout = TimeSpan.FromMilliseconds(500);
        for (int i = 0; i < totalRequests; i++)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            try
            {
                var res = await client.GetAsync("http://localhost:8080/health");
                sw.Stop();
                if (res.IsSuccessStatusCode)
                {
                    totalMs += sw.ElapsedMilliseconds;
                    successCount++;
                }
            }
            catch
            {
                sw.Stop();
            }
        }
    }

    if (successCount < 95)
    {
        ExecuteTerminalCommand($"docker rm -f {containerName}");
        SaveLifecycleLog(cycleId, "StressTestFailed", $"Stage 2 failed. Request success rate was only {successCount}% (required: 95%).");
        return new SandboxTestResult(false, $"Uji Validitas GAGAL: Success rate was only {successCount}% under stress test.", 0, 0);
    }

    long avgLatencyMs = totalMs / successCount;
    
    // Log Stage 2 Success
    SaveLifecycleLog(cycleId, "StressTestSuccess", $"Stage 2 passed. Stress test completed with success rate: {successCount}%. Average latency: {avgLatencyMs} ms.");

    // Log Stage 3: RAM Check
    SaveLifecycleLog(cycleId, "RamCheckStart", "Stage 3: Analyzing RAM consumption of shadow container using docker stats.");

    // Uji RAM: Read resource usage using docker stats
    long ramUsageBytes = 0;
    string statsOutput = ExecuteTerminalCommand($"docker stats --no-stream --format \"{{{{.MemUsage}}}}\" {containerName}");
    if (!string.IsNullOrEmpty(statsOutput))
    {
        ramUsageBytes = ParseMemoryUsage(statsOutput);
    }

    // Terminate shadow container
    ExecuteTerminalCommand($"docker rm -f {containerName}");

    // Benchmark comparison: read current production average latency (simulate baseline)
    long baselineLatencyMs = 8; // baseline benchmark fallback
    long latencyDeltaMs = avgLatencyMs - baselineLatencyMs;

    // Punishment Protocol: Dynamic RAM comparison vs production baseline (NOT a hardcoded static limit)
    double mutantRamMb = ramUsageBytes / (1024.0 * 1024.0);
    double toleranceMb = baselineRamMb > 0 ? baselineRamMb * 1.10 : 250.0; // baseline + 10% tolerance, or 250MB fallback
    long maxAllowedRam = (long)(toleranceMb * 1024 * 1024);

    if (ramUsageBytes > 0 && ramUsageBytes > maxAllowedRam)
    {
        double excessMb = mutantRamMb - (baselineRamMb > 0 ? baselineRamMb : toleranceMb);
        double wastePercent = baselineRamMb > 0 ? (excessMb / baselineRamMb * 100.0) : 0;
        string punishmentMsg = $"PUNISHMENT REPORT — Mutasi GAGAL.\n" +
                               $"Mutasimu memakan RAM {mutantRamMb:F1}MB, sedangkan baseline produksi {(baselineRamMb > 0 ? baselineRamMb.ToString("F1") : "?")}MB.\n" +
                               $"Batas toleransi yang dilanggar: {toleranceMb:F1}MB (baseline + 10%).\n" +
                               $"Pemborosan: +{excessMb:F1}MB ({(baselineRamMb > 0 ? wastePercent.ToString("F0") : "?")}%) lebih boros dari versi produksi saat ini.\n" +
                               $"Pada siklus berikutnya, kamu WAJIB mengurangi konsumsi memori, bukan menambahnya.";
        
        SavePunishmentLog(baselineRamMb, mutantRamMb, excessMb, 0, punishmentMsg);
        SaveLifecycleLog(cycleId, "RamCheckFailed", $"Stage 3 failed (Punished). RAM usage: {mutantRamMb:F1}MB exceeds limit: {toleranceMb:F1}MB (baseline: {(baselineRamMb > 0 ? baselineRamMb.ToString("F1") : "?")}MB + 10%).");
        Console.WriteLine($"[Darwinian Sandbox] 🚨 PUNISHMENT PROTOCOL: {punishmentMsg}");
        return new SandboxTestResult(false, $"Uji RAM GAGAL (Punishment Protocol): {punishmentMsg}", latencyDeltaMs, ramUsageBytes);
    }

    // Log Stage 3 Success
    SaveLifecycleLog(cycleId, "RamCheckSuccess", $"Stage 3 passed. RAM usage: {mutantRamMb:F1}MB is within safe limit: {toleranceMb:F1}MB (baseline: {(baselineRamMb > 0 ? baselineRamMb.ToString("F1") : "?")}MB + 10%).");

    return new SandboxTestResult(true, "Lulus Semua Uji Kelayakan", latencyDeltaMs, ramUsageBytes);
}

// Execute terminal command helper
string ExecuteTerminalCommand(string command)
{
    try
    {
        var process = new System.Diagnostics.Process
        {
            StartInfo = new System.Diagnostics.ProcessStartInfo
            {
                FileName = "/bin/sh",
                Arguments = $"-c \"{command}\"",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            }
        };
        process.Start();
        string output = process.StandardOutput.ReadToEnd();
        string error = process.StandardError.ReadToEnd();
        process.WaitForExit();
        
        return string.IsNullOrEmpty(error) ? output.Trim() : $"{output}\nError: {error}".Trim();
    }
    catch (Exception ex)
    {
        return $"Error executing command: {ex.Message}";
    }
}

// Parse memory usage from docker stats output (e.g. "120.5MiB / 7.7GiB")
long ParseMemoryUsage(string statsOutput)
{
    try
    {
        var match = Regex.Match(statsOutput, @"([\d\.]+)\s*(MiB|GiB|B|KB|MB|GB|KiB)", RegexOptions.IgnoreCase);
        if (match.Success)
        {
            double val = double.Parse(match.Groups[1].Value);
            string unit = match.Groups[2].Value.ToLower();
            if (unit.Contains("g")) return (long)(val * 1024 * 1024 * 1024);
            if (unit.Contains("m")) return (long)(val * 1024 * 1024);
            if (unit.Contains("k")) return (long)(val * 1024);
            return (long)val;
        }
    }
    catch { }
    return 0;
}

// Get Container RAM resources from cgroup
(long usage, long limit) GetMemoryStats()
{
    long usage = 0;
    long limit = 0;
    try
    {
        if (File.Exists("/sys/fs/cgroup/memory.current"))
        {
            long.TryParse(File.ReadAllText("/sys/fs/cgroup/memory.current").Trim(), out usage);
        }
        if (File.Exists("/sys/fs/cgroup/memory.max"))
        {
            string limitStr = File.ReadAllText("/sys/fs/cgroup/memory.max").Trim();
            if (limitStr != "max")
            {
                long.TryParse(limitStr, out limit);
            }
        }
        if (usage == 0 && File.Exists("/sys/fs/cgroup/memory/memory.usage_in_bytes"))
        {
            long.TryParse(File.ReadAllText("/sys/fs/cgroup/memory/memory.usage_in_bytes").Trim(), out usage);
        }
        if (limit == 0 && File.Exists("/sys/fs/cgroup/memory/memory.limit_in_bytes"))
        {
            long.TryParse(File.ReadAllText("/sys/fs/cgroup/memory/memory.limit_in_bytes").Trim(), out limit);
        }
    }
    catch (Exception ex)
    {
        Console.WriteLine($"[Doctor Memory Error] {ex.Message}");
    }
    return (usage, limit);
}

// Get Container Storage stats from DriveInfo on the shared volume
(double freeGb, double totalGb) GetStorageStats()
{
    try
    {
        string targetPath = Directory.Exists("/app_host") ? "/app_host" : "/";
        var drive = new DriveInfo(targetPath);
        double freeGb = drive.AvailableFreeSpace / (1024.0 * 1024.0 * 1024.0);
        double totalGb = drive.TotalSize / (1024.0 * 1024.0 * 1024.0);
        return (freeGb, totalGb);
    }
    catch (Exception ex)
    {
        Console.WriteLine($"[Doctor Storage Error] {ex.Message}");
        return (0, 0);
    }
}

// Get full resource snapshot combining RAM (cgroup) + Storage (DriveInfo) with status classification
ResourceSnapshot GetResourceSnapshot()
{
    var (memUsage, memLimit) = GetMemoryStats();
    double ramUsedMb = memUsage / (1024.0 * 1024.0);
    double ramLimitMb = memLimit / (1024.0 * 1024.0);
    double ramFreeMb = ramLimitMb > 0 ? ramLimitMb - ramUsedMb : 0;
    double ramUsagePct = (ramLimitMb > 0 && ramUsedMb > 0) ? (ramUsedMb / ramLimitMb) * 100.0 : 0;

    string ramStatus;
    if (ramLimitMb <= 0)        ramStatus = "AMAN";    // No cgroup limit, assume safe
    else if (ramUsagePct >= 85) ramStatus = "KRITIS";
    else if (ramUsagePct >= 70) ramStatus = "WASPADA";
    else                        ramStatus = "AMAN";

    if (ramLimitMb <= 0) ramFreeMb = 1024; // Fallback assumption: 1GB free when no limit is set

    var (storageFreeGb, storageTotalGb) = GetStorageStats();
    string storageStatus = (storageFreeGb == 0) ? "AMAN" // Unknown, assume safe
                         : storageFreeGb < 2.0  ? "KRITIS"
                         : storageFreeGb < 5.0  ? "WASPADA"
                         : "AMAN";

    return new ResourceSnapshot(ramUsedMb, ramLimitMb, ramFreeMb, ramUsagePct, storageFreeGb, storageTotalGb, ramStatus, storageStatus);
}

// Build dynamic resource-aware Survival Prompt for Ollama
string BuildSurvivalPrompt(ResourceSnapshot snapshot, string lastPunishment, string baseInstruction, string codeSnippet, string filePath)
{
    string mandate;
    if (snapshot.RamStatus == "KRITIS")
        mandate = "- ⚠️ STATUS RAM KRITIS: Prioritas MUTLAK adalah mencari dan menghapus memory leak. DILARANG menambahkan library baru atau membuat cache besar di disk.";
    else if (snapshot.RamStatus == "WASPADA")
        mandate = "- ⚠️ STATUS RAM WASPADA: Optimasi efisiensi memori. Boleh refactor struktur data. DILARANG menambahkan dependency Python baru.";
    else
        mandate = "- ✅ STATUS RAM AMAN: Bebas optimasi. Boleh refactor algoritmik atau tambah fitur ringan. Jangan tingkatkan konsumsi RAM secara signifikan.";

    string storageMandate = "";
    if (snapshot.StorageStatus == "KRITIS")
        storageMandate = "\n- ⚠️ STATUS STORAGE KRITIS: DILARANG membuat file cache baru atau log berukuran besar di disk.";
    else if (snapshot.StorageStatus == "WASPADA")
        storageMandate = "\n- ⚠️ STATUS STORAGE WASPADA: Minimalisir penggunaan disk. Jangan buat file temporary besar.";

    string punishmentBlock = string.IsNullOrEmpty(lastPunishment) ? "" :
        $"\n\n⚠️ LAPORAN HUKUMAN DARI SIKLUS MUTASI SEBELUMNYA (PELAJARI INI):\n{lastPunishment}\nKesalahan di atas TIDAK BOLEH diulangi pada mutasi ini.";

    string ramLimitStr = snapshot.RamLimitMb > 0 ? $"{snapshot.RamLimitMb:F0}MB" : "tidak dibatasi";
    double maxAllowedMb = snapshot.RamUsedMb * 1.10;

    string evolutionHistory = GetRecentEvolutionHistory();
    string historyBlock = $"\n\n📜 RIWAYAT EVOLUSI TERKINI (PELAJARI SIKLUS SEBELUMNYA AGAR LEBIH PINTAR):\n{evolutionHistory}";

    return $"Sistem saat ini idle. Ini adalah kode FastAPI terkini milikmu. Evaluasi dan optimalkan.\n\n" +
           $"📊 KONDISI SUMBER DAYA REAL-TIME:\n" +
           $"- RAM Terpakai: {snapshot.RamUsedMb:F0}MB / {ramLimitStr} ({snapshot.RamUsagePct:F0}% — Status: {snapshot.RamStatus})\n" +
           $"- Sisa RAM Aman: {snapshot.RamFreeMb:F0}MB\n" +
           $"- Sisa Storage: {snapshot.StorageFreeGb:F1}GB / {snapshot.StorageTotalGb:F1}GB (Status: {snapshot.StorageStatus})\n\n" +
           $"🎯 MANDATMU berdasarkan kondisi sumber daya di atas:\n" +
           $"{mandate}{storageMandate}\n" +
           $"- 🚫 BATASAN MUTASI: Kode barumu TIDAK BOLEH menggunakan RAM melebihi {snapshot.RamUsedMb:F0}MB + toleransi 10% = {maxAllowedMb:F0}MB." +
           $"{punishmentBlock}" +
           $"{historyBlock}\n\n" +
           $"📋 INSTRUKSI SPESIFIK: {baseInstruction}\n\n" +
           $"Berikut adalah kode Python yang perlu kamu optimalkan (file: {filePath}):\n\n```python\n{codeSnippet}\n```";
}

// Save punishment log to MySQL for cross-cycle learning persistence
void SavePunishmentLog(double ramBaselineMb, double ramMutantMb, double ramExcessMb, double storageFreeGb, string punishmentMessage)
{
    try
    {
        var connStr = GetMySqlConnectionString();
        using var conn = new MySqlConnection(connStr);
        conn.Open();

        int cycleNumber = 1;
        using (var countCmd = conn.CreateCommand())
        {
            countCmd.CommandText = "SELECT COUNT(*) FROM punishment_logs";
            cycleNumber = Convert.ToInt32(countCmd.ExecuteScalar()) + 1;
        }

        using var cmd = conn.CreateCommand();
        cmd.CommandText = @"INSERT INTO punishment_logs
            (id, cycle_number, ram_baseline_mb, ram_mutant_mb, ram_excess_mb, storage_free_gb, punishment_message, timestamp)
            VALUES (@id, @cycle, @baseline, @mutant, @excess, @storage, @message, @timestamp)";
        cmd.Parameters.AddWithValue("@id", Guid.NewGuid().ToString());
        cmd.Parameters.AddWithValue("@cycle", cycleNumber);
        cmd.Parameters.AddWithValue("@baseline", ramBaselineMb);
        cmd.Parameters.AddWithValue("@mutant", ramMutantMb);
        cmd.Parameters.AddWithValue("@excess", ramExcessMb);
        cmd.Parameters.AddWithValue("@storage", storageFreeGb);
        cmd.Parameters.AddWithValue("@message", punishmentMessage);
        cmd.Parameters.AddWithValue("@timestamp", DateTime.UtcNow);
        cmd.ExecuteNonQuery();

        Console.WriteLine($"[Doctor Punishment] 💾 Punishment #{cycleNumber} saved: baseline {ramBaselineMb:F1}MB → mutant {ramMutantMb:F1}MB (excess: +{ramExcessMb:F1}MB)");
    }
    catch (Exception ex)
    {
        Console.WriteLine($"[Doctor Punishment Error] Failed to save punishment log: {ex.Message}");
    }
}

// Get the last punishment message to inject into the next evolution cycle
string GetLastPunishmentMessage()
{
    try
    {
        var connStr = GetMySqlConnectionString();
        using var conn = new MySqlConnection(connStr);
        conn.Open();

        using var cmd = conn.CreateCommand();
        cmd.CommandText = "SELECT punishment_message FROM punishment_logs ORDER BY timestamp DESC LIMIT 1";
        var result = cmd.ExecuteScalar();
        return result?.ToString() ?? "";
    }
    catch
    {
        return "";
    }
}

// Get the last 5 evolution cycles to teach the AI what worked and what failed
string GetRecentEvolutionHistory()
{
    try
    {
        var connStr = GetMySqlConnectionString();
        using var conn = new MySqlConnection(connStr);
        conn.Open();

        using var cmd = conn.CreateCommand();
        cmd.CommandText = "SELECT timestamp, status, latency_change_ms, ram_usage_bytes, failure_reason FROM evolution_logs ORDER BY timestamp DESC LIMIT 5";

        var sb = new System.Text.StringBuilder();
        using var reader = cmd.ExecuteReader();
        int count = 1;
        while (reader.Read())
        {
            var ts = reader.GetDateTime(0);
            var status = reader.GetString(1);
            var latency = reader.GetInt32(2);
            var ramBytes = reader.GetInt64(3);
            var reason = reader.IsDBNull(4) ? "" : reader.GetString(4);

            double ramMb = ramBytes / (1024.0 * 1024.0);

            sb.AppendLine($"  {count}. [{ts:yyyy-MM-dd HH:mm:ss}] Status: {status}");
            if (status == "Lulus")
            {
                sb.AppendLine($"     Perubahan Latensi: {latency}ms | RAM Mutan: {ramMb:F1}MB");
            }
            else
            {
                sb.AppendLine($"     Penyebab Gagal: {reason}");
            }
            count++;
        }

        if (sb.Length == 0)
        {
            return "Belum ada riwayat evolusi sebelumnya.";
        }
        return sb.ToString();
    }
    catch (Exception ex)
    {
        return $"Gagal mengambil riwayat evolusi: {ex.Message}";
    }
}


// Build a targeted error-repair prompt for Ollama based on actual FastAPI ERROR log lines
string BuildErrorRepairPrompt(string errorBlock)
{
    var snapshot = GetResourceSnapshot();
    string lastPunishment = GetLastPunishmentMessage();
    string punishmentBlock = string.IsNullOrEmpty(lastPunishment) ? "" :
        $"\n\n⚠️ CATATAN: Pada siklus mutasi sebelumnya kamu dihukum karena:\n{lastPunishment}\nJangan ulangi kesalahan yang sama.";

    string evolutionHistory = GetRecentEvolutionHistory();
    string historyBlock = $"\n\n📜 RIWAYAT EVOLUSI TERKINI (PELAJARI SIKLUS SEBELUMNYA AGAR LEBIH PINTAR):\n{evolutionHistory}";

    return $"🚨 DARURAT: FastAPI mengalami ERROR aktif yang perlu segera diperbaiki.\n\n" +
           $"📋 LOG ERROR DARI FASTAPI:\n```\n{errorBlock}\n```\n\n" +
           $"📊 KONDISI SUMBER DAYA SAAT INI:\n" +
           $"- RAM: {snapshot.RamUsedMb:F0}MB ({snapshot.RamStatus}) | Storage: {snapshot.StorageFreeGb:F1}GB ({snapshot.StorageStatus})\n\n" +
           $"🎯 MANDATMU:\n" +
           $"1. Analisis akar penyebab (root cause) dari error di atas.\n" +
           $"2. Perbaiki kode Python FastAPI yang menyebabkan error tersebut.\n" +
           $"3. Tambahkan error handling yang lebih baik agar error serupa tidak terulang.\n" +
           $"4. JANGAN ubah logika bisnis utama — hanya perbaiki bagian yang error.\n" +
           $"5. Pastikan kode hasil perbaikan TIDAK meningkatkan konsumsi RAM lebih dari {snapshot.RamUsedMb * 1.10:F0}MB." +
           $"{punishmentBlock}" +
           $"{historyBlock}";
}

// MySQL code backups
void SaveBackupToDB(string relativePath, string fullPath, string content, string backupsDir)
{
    try
    {
        Directory.CreateDirectory(backupsDir);
        string backupFileName = $"{Path.GetFileName(fullPath)}.{DateTime.UtcNow.Ticks}.bak";
        string backupFullPath = Path.Combine(backupsDir, backupFileName);
        
        File.WriteAllText(backupFullPath, content);
        
        var connStr = GetMySqlConnectionString();
        using var conn = new MySqlConnection(connStr);
        conn.Open();
        
        using var cmd = conn.CreateCommand();
        cmd.CommandText = "INSERT INTO code_backups (id, file_path, backup_file_name, timestamp, status) VALUES (@id, @file_path, @backup_file_name, @timestamp, @status)";
        cmd.Parameters.AddWithValue("@id", Guid.NewGuid().ToString());
        cmd.Parameters.AddWithValue("@file_path", relativePath);
        cmd.Parameters.AddWithValue("@backup_file_name", backupFileName);
        cmd.Parameters.AddWithValue("@timestamp", DateTime.UtcNow);
        cmd.Parameters.AddWithValue("@status", "Active");
        
        cmd.ExecuteNonQuery();
        Console.WriteLine($"[Doctor DB] Recorded backup for {relativePath} in MySQL database.");
    }
    catch (Exception ex)
    {
        Console.WriteLine($"[Doctor DB Error] SaveBackupToDB failed: {ex.Message}");
    }
}

// Rollback the last active backup
async Task RollbackLastBackupAsync(string rootDir, string dbPath, string backupsDir)
{
    try
    {
        var connStr = GetMySqlConnectionString();
        using var conn = new MySqlConnection(connStr);
        conn.Open();
        
        string id = "";
        string filePath = "";
        string backupFileName = "";
        
        using (var cmd = conn.CreateCommand())
        {
            cmd.CommandText = "SELECT id, file_path, backup_file_name FROM code_backups WHERE status = 'Active' ORDER BY timestamp DESC LIMIT 1";
            using var reader = cmd.ExecuteReader();
            if (reader.Read())
            {
                id = reader.GetString(0);
                filePath = reader.GetString(1);
                backupFileName = reader.GetString(2);
            }
        }

        if (string.IsNullOrEmpty(backupFileName))
        {
            Console.WriteLine("[Doctor Rollback] No active backups found in MySQL DB.");
            return;
        }

        string backupFile = Path.Combine(backupsDir, backupFileName);
        string targetFile = Path.Combine(rootDir, filePath);

        if (File.Exists(backupFile))
        {
            string oldContent = File.ReadAllText(backupFile);
            File.WriteAllText(targetFile, oldContent);
            
            using (var cmd = conn.CreateCommand())
            {
                cmd.CommandText = "UPDATE code_backups SET status = 'RolledBack' WHERE id = @id";
                cmd.Parameters.AddWithValue("@id", id);
                cmd.ExecuteNonQuery();
            }
            
            Console.WriteLine($"[Doctor Rollback] Rolled back {filePath} to backup {backupFileName}!");
        }
    }
    catch (Exception ex)
    {
        Console.WriteLine($"[Doctor Rollback Error] Rollback failed: {ex.Message}");
    }
}

// MySQL Checkpoint operations (Pause and Continue)
CheckpointRecord? GetLatestCheckpoint()
{
    try
    {
        var connStr = GetMySqlConnectionString();
        using var conn = new MySqlConnection(connStr);
        conn.Open();
        
        using var cmd = conn.CreateCommand();
        cmd.CommandText = "SELECT id, task_description, status, timestamp FROM checkpoints WHERE status = 'Paused' ORDER BY timestamp DESC LIMIT 1";
        using var reader = cmd.ExecuteReader();
        if (reader.Read())
        {
            return new CheckpointRecord(
                reader.GetString(0),
                reader.GetString(1),
                reader.GetString(2),
                reader.GetDateTime(3)
            );
        }
    }
    catch { }
    return null;
}

void SaveCheckpoint(string taskDescription, string status)
{
    try
    {
        var connStr = GetMySqlConnectionString();
        using var conn = new MySqlConnection(connStr);
        conn.Open();
        
        using var cmd = conn.CreateCommand();
        cmd.CommandText = "INSERT INTO checkpoints (id, task_description, status, timestamp) VALUES (@id, @task, @status, @timestamp)";
        cmd.Parameters.AddWithValue("@id", Guid.NewGuid().ToString());
        cmd.Parameters.AddWithValue("@task", taskDescription);
        cmd.Parameters.AddWithValue("@status", status);
        cmd.Parameters.AddWithValue("@timestamp", DateTime.UtcNow);
        
        cmd.ExecuteNonQuery();
        Console.WriteLine($"[Doctor Checkpoint] Checkpoint saved: {taskDescription} (Status: {status})");
    }
    catch (Exception ex)
    {
        Console.WriteLine($"[Doctor Checkpoint Error] Save failed: {ex.Message}");
    }
}

void ClearCheckpoints()
{
    try
    {
        var connStr = GetMySqlConnectionString();
        using var conn = new MySqlConnection(connStr);
        conn.Open();
        
        using var cmd = conn.CreateCommand();
        cmd.CommandText = "DELETE FROM checkpoints";
        cmd.ExecuteNonQuery();
    }
    catch { }
}

// MySQL evolution log recording
void SaveEvolutionLog(string filePath, string fromVersion, string toVersion, string mutatedCode, long latencyDeltaMs, long ramUsageBytes, string status, string failureReason)
{
    try
    {
        var connStr = GetMySqlConnectionString();
        using var conn = new MySqlConnection(connStr);
        conn.Open();
        
        using var cmd = conn.CreateCommand();
        cmd.CommandText = @"INSERT INTO evolution_logs 
            (id, file_path, from_version, to_version, mutated_code, latency_change_ms, ram_usage_bytes, status, failure_reason, timestamp) 
            VALUES 
            (@id, @file_path, @from, @to, @code, @latency, @ram, @status, @reason, @timestamp)";
            
        cmd.Parameters.AddWithValue("@id", Guid.NewGuid().ToString());
        cmd.Parameters.AddWithValue("@file_path", filePath);
        cmd.Parameters.AddWithValue("@from", fromVersion);
        cmd.Parameters.AddWithValue("@to", toVersion);
        cmd.Parameters.AddWithValue("@code", mutatedCode);
        cmd.Parameters.AddWithValue("@latency", (int)latencyDeltaMs);
        cmd.Parameters.AddWithValue("@ram", ramUsageBytes);
        cmd.Parameters.AddWithValue("@status", status);
        cmd.Parameters.AddWithValue("@reason", failureReason);
        cmd.Parameters.AddWithValue("@timestamp", DateTime.UtcNow);
        
        cmd.ExecuteNonQuery();
        Console.WriteLine($"[Doctor DB] Evolution logged. Status: {status}, File: {filePath}");
    }
    catch (Exception ex)
    {
        Console.WriteLine($"[Doctor DB Error] Failed to log evolution: {ex.Message}");
    }
}

// MySQL lifecycle event logging
void SaveLifecycleLog(string cycleId, string eventType, string details)
{
    try
    {
        var connStr = GetMySqlConnectionString();
        using var conn = new MySqlConnection(connStr);
        conn.Open();
        
        using var cmd = conn.CreateCommand();
        cmd.CommandText = @"INSERT INTO doctor_lifecycle_logs 
            (id, cycle_id, event_type, details, timestamp) 
            VALUES 
            (@id, @cycle_id, @event_type, @details, @timestamp)";
            
        cmd.Parameters.AddWithValue("@id", Guid.NewGuid().ToString());
        cmd.Parameters.AddWithValue("@cycle_id", cycleId);
        cmd.Parameters.AddWithValue("@event_type", eventType);
        cmd.Parameters.AddWithValue("@details", details);
        cmd.Parameters.AddWithValue("@timestamp", DateTime.UtcNow);
        
        cmd.ExecuteNonQuery();
        Console.WriteLine($"[Doctor Lifecycle] [{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss}] {eventType}: {details}");
    }
    catch (Exception ex)
    {
        Console.WriteLine($"[Doctor Lifecycle Error] Failed to log lifecycle event: {ex.Message}");
    }
}

public record RequestContext(string Method, string Path, Dictionary<string, string> Headers, string Body);
public class MiddlewareResponse {
    public string Action { get; set; } = "allow";
    public int StatusCode { get; set; } = 200;
    public string Detail { get; set; } = "";
    public Dictionary<string, string>? ModifiedHeaders { get; set; }
    public string? ModifiedBody { get; set; }
}

public record UpgradeRequest(string FilePath, string Action, string SearchContent, string Content, bool TriggerRebuild);
public record AnalyzeRequest(string? FilePath, string? CustomPrompt);
public class UpgradeResponse {
    public bool Success { get; set; }
    public string Error { get; set; } = "";
}

public record CheckpointRecord(string Id, string TaskDescription, string Status, DateTime Timestamp);
public record SandboxTestResult(bool Lulus, string Reason, long LatencyDeltaMs, long RamUsageBytes);
public record ResourceSnapshot(
    double RamUsedMb, double RamLimitMb, double RamFreeMb, double RamUsagePct,
    double StorageFreeGb, double StorageTotalGb,
    string RamStatus, string StorageStatus
);
