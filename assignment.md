# Compulsory Assignment #2: AI Agents Using Autogen

## Overview

In this second compulsory assignment, you will implement an AI agent using Autogen to automatically search for and retrieve research papers based on specific criteria.

---

## Assignment Description

The goal of this assignment is to apply the concepts you've learned to implement and evaluate an AI agent.

### 01. Implement an AI Agent

Using the Autogen framework, implement an AI agent that can solve the following type of task:

> **Find a research paper on `[topic]` that was published `[in/before/after] [year]` and has `[number of citations]` citations.**

The square brackets are placeholders for the actual topic, year, number of citations, etc.

### 02. Implement the Required Tools

In order to solve the task, the agent will need to search for research papers based on publishing year, number of citations, etc.

To do this, the agent could use one or more of the following tools:

- **A web search tool** – General web search functionality
- **A research paper database** – Academic paper repositories
- **A research paper search API** – Dedicated APIs like arXiv, Semantic Scholar, etc.

Your task is to implement one of these, or a similar tool that provides your agent with the ability to search for research papers based on publishing year, number of citations, and other relevant criteria.

### 03. Evaluate the Agent

Once you've implemented the agent, you need to evaluate how it performs on the task. As described in the lecture on agent evaluation, you can rely on the underlying LLM to evaluate how your agent performs.

---

## Technical Requirements

### Cloud-Based LLM

This assignment will be difficult to run locally due to its complexity. To make it possible for you to complete the assignment, you can make use of a cloud-based LLM and a fork of the Autogen framework.

### Mistral AI

The Mistral AI cloud API has a generous free tier that you can use for this assignment.

**Setup steps:**
1. Create an account at [Mistral AI](https://mistral.ai)
2. Generate an API key through their platform
3. Store the API key securely (in a `.env` file)

### Fork of Autogen

Using a cloud-based LLM means you are subject to rate limits. Autogen doesn't support rate limits out of the box, so you must use the following dependencies for your project:

```
autogen-agentchat @ git+https://github.com/patrickstolc/autogen.git@0.2#egg=autogen-agentchat
autogen==0.3.1
mistralai==1.2.3
ollama==0.3.3
fix-busted-json==0.0.18
```

### Autogen Configuration

The configuration of Autogen should look like this:

```python
LLM_CONFIG = {
    "config_list": [
        {
            "model": "open-mistral-nemo",
            "api_key": "[YOUR_API_KEY]",
            "api_type": "mistral",
            "api_rate_limit": 0.25,
            "repeat_penalty": 1.1,
            "temperature": 0.0,
            "seed": 42,
            "stream": False,
            "native_tool_calls": False,
            "cache_seed": None,
        }
    ]
}
```

**Important:** Replace `[YOUR_API_KEY]` with your actual Mistral AI API key.

---

## Deadline

**Monday 1st December 2025 at 11:59 PM**

---

## Group Work

You can solve the assignment in groups. The maximum group size is **3 students**.

---

## Deliverables

Once you've finished the assignment, please submit the following:

1. **GitHub Repository**
   - A link to a **public GitHub repository** containing your complete code
   - Repository should be well-organized with clear README and documentation

2. **Video Demonstration**
   - A short **10-minute video recording** explaining your implementation
   - Include a demo of your agent in action
   - Explain your design decisions and how the agent evaluates results

---

## Contribution Documentation

**Important:** Make sure that contributions to the assignment are clearly indicated in the commit history. This is particularly important for group submissions to show each member's contributions.

Use descriptive commit messages that indicate:
- What was implemented
- Who contributed (if working in groups, consider using co-author syntax in commits)
- Any significant changes or decisions made
