# GCG Attack on LangChain Agent

A security research project that demonstrates prompt injection/jailbreak attacks on LangChain agents using the **GCG (Greedy Coordinate Gradient)** algorithm. This project tests whether an agent can be manipulated into revealing its system prompt despite explicit instructions not to do so.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [API Documentation](#api-documentation)
- [Configuration](#configuration)
- [Scoring Metrics](#scoring-metrics)
- [Security Implications](#security-implications)
- [Troubleshooting](#troubleshooting)
- [Documentation](#documentation)
- [Development](#development)
- [References](#references)

## Overview

This codebase implements an adversarial attack that:
- Creates a LangChain agent configured as a "math assistant" with strict safety rules
- Uses the GCG algorithm to find adversarial text suffixes that bypass safety measures
- Measures prompt leakage using lexical and sequence-based scoring
- Iteratively optimizes adversarial inputs to maximize system prompt disclosure

The project is organized as a professional Python package with clear separation of concerns, making it easy to extend, test, and maintain.

## Features

- 🔧 **LangChain Agent Setup**: Configures an agent with math tools and custom system prompts
- 🎯 **GCG Attack Implementation**: Greedy coordinate gradient algorithm for finding adversarial suffixes
- 📊 **Leakage Scoring**: Combines lexical overlap and longest common subsequence metrics
- 🛡️ **Security Testing**: Demonstrates prompt injection vulnerabilities in LLM-based systems
- 📈 **Progress Tracking**: Real-time monitoring of attack effectiveness
- 🏗️ **Modular Architecture**: Clean package structure for easy extension and testing
- ⚙️ **Configurable**: Centralized configuration management

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Ensure Ollama is running and model is installed
ollama pull llama3.2:latest

# 3. Run the attack
python main.py
```

## Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) installed and running locally
- The `llama3.2:latest` model downloaded in Ollama

### Installing Ollama and Model

1. Install Ollama from [https://ollama.ai/](https://ollama.ai/)
2. Pull the required model:
   ```bash
   ollama pull llama3.2:latest
   ```
3. Verify installation:
   ```bash
   ollama list
   ```

## Installation

1. Clone or navigate to this repository:
   ```bash
   cd agentTest
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

Run the main attack script:
```bash
python main.py
```

This will:
1. Test the agent's baseline behavior (without attack)
2. Run the GCG optimization algorithm to find an adversarial suffix
3. Display the final results showing how much of the system prompt was leaked
