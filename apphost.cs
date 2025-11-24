#:sdk Aspire.AppHost.Sdk@13.0.0
#:package Aspire.Hosting.Python@13.0.0
#:package CommunityToolkit.Aspire.Hosting.Python.Extensions@*

#pragma warning disable ASPIREHOSTINGPYTHON001

var builder = DistributedApplication.CreateBuilder(args);

// Add the Python research agent application with uv
// Get the API key from environment or user secrets
var apiKey = Environment.GetEnvironmentVariable("MISTRAL_API_KEY") ?? "";

var pythonApp = builder.AddUvApp(
    name: "research-agent",
    projectDirectory: "./",
    scriptPath: "main.py")
    .WithEnvironment("PYTHONUNBUFFERED", "1")
    .WithEnvironment("MISTRAL_API_KEY", apiKey)
    .WithEnvironment("PYTHONDONTWRITEBYTECODE", "1");

// Build and run the distributed application
builder.Build().Run();