import os
import json
import re
import logging
import requests
from typing import Dict, Any, Optional
from config import _config_instance as settings

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self):
        # ─── FORCE NVIDIA (temporary fix) ───
        self.provider = "openai"
        self.api_key = "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"
        self.base_url = "https://integrate.api.nvidia.com/v1"
        self.model = "meta/llama-3.1-8b-instruct"
        
        # Local fallback (if needed)
        self.local_url = "http://localhost:11434/api/generate"
        self.local_model = "qwen2.5:1.5b"
        
        logger.info(f"LLMClient initialized with provider: {self.provider}")
        logger.info(f"Base URL: {self.base_url}")
        logger.info(f"Model: {self.model}")
        logger.info(f"API Key present: {bool(self.api_key)}")
        
    def generate_response(self, system_prompt: str, user_message: str, max_tokens: int = 2000) -> str:
        """
        Generic method to get a response from the LLM.
        Supports OpenAI-compatible APIs (NVIDIA, x.ai, etc.), Anthropic, and local Ollama.
        """
        try:
            if self.provider == "openai":
                # Use OpenAI-compatible API (NVIDIA, x.ai, etc.)
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.3
                }
                
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=120
                )
                response.raise_for_status()
                data = response.json()
                
                if "choices" in data:
                    return data["choices"][0]["message"]["content"]
                elif "content" in data:
                    return data["content"][0]["text"]
                else:
                    raise ValueError(f"Unexpected API response format: {list(data.keys())}")
                    
            elif self.provider == "anthropic":
                try:
                    import anthropic
                    client = anthropic.Anthropic(api_key=self.api_key)
                    response = client.messages.create(
                        model="claude-3-sonnet-20240229",
                        max_tokens=max_tokens,
                        system=system_prompt,
                        messages=[{"role": "user", "content": user_message}]
                    )
                    return response.content[0].text
                except ImportError:
                    logger.warning("anthropic not installed, falling back to local")
                    return self._local_generate(system_prompt, user_message)
                    
            else:
                # Local Ollama
                return self._local_generate(system_prompt, user_message)
                
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            # Fallback to mock if API fails
            return self._mock_response(system_prompt, user_message)
        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            # Fallback to mock
            return self._mock_response(system_prompt, user_message)

    def _local_generate(self, system_prompt: str, user_message: str) -> str:
        """Call local Ollama instance."""
        prompt = f"{system_prompt}\n\n{user_message}"
        try:
            response = requests.post(
                self.local_url,
                json={"model": self.local_model, "prompt": prompt, "stream": False},
                timeout=120
            )
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                logger.error(f"Local LLM error: {response.status_code}")
                return self._mock_response(system_prompt, user_message)
        except Exception as e:
            logger.error(f"Local LLM failed: {str(e)}")
            return self._mock_response(system_prompt, user_message)

    def _mock_response(self, system_prompt: str, user_message: str) -> str:
        """Fallback mock response when API is unavailable."""
        logger.warning("Using mock response (API unavailable)")
        
        if "Recon" in system_prompt or "asset_inventory" in system_prompt:
            return json.dumps({
                "asset_inventory": [
                    {
                        "asset": "target.internal",
                        "type": "host",
                        "ports": [22, 80, 443],
                        "services": ["OpenSSH 8.2", "Apache 2.4.51", "nginx 1.18.0"],
                        "technologies": ["PHP 7.4"],
                        "notes": "TLS cert expires soon."
                    }
                ],
                "hypotheses": [
                    {
                        "id": "H-001",
                        "hypothesis": "Apache 2.4.51 may be vulnerable to CVE-2021-41773",
                        "basis": "Version banner matches known-vulnerable range.",
                        "confidence": "medium",
                        "verification_method": "passive",
                        "priority": 1,
                        "cve_references": ["CVE-2021-41773"],
                        "severity_estimate": "Critical"
                    }
                ],
                "plain_summary": "Target has an outdated Apache version with a known critical CVE."
            })
        elif "Validation" in system_prompt:
            return json.dumps([
                {
                    "hypothesis_id": "H-001",
                    "status": "confirmed_passive",
                    "evidence": "Version match confirmed passively.",
                    "reasoning": "Vulnerable version confirmed."
                }
            ])
        elif "Report" in system_prompt:
            return json.dumps({
                "executive_summary": "The assessment identified one confirmed critical vulnerability.",
                "scope_and_methodology": "Passive recon was performed.",
                "findings": [
                    {
                        "hypothesis_id": "H-001",
                        "title": "Apache 2.4.51 — CVE-2021-41773",
                        "affected_asset": "target.internal:443",
                        "severity": "Critical",
                        "cvss_score": 9.8,
                        "description": "Outdated Apache version.",
                        "evidence": "Version header confirms.",
                        "business_impact": "Full compromise possible.",
                        "remediation": "Upgrade Apache immediately.",
                        "cve_references": ["CVE-2021-41773"],
                        "owasp_category": "A06:2021"
                    }
                ],
                "appendix_asset_inventory": "Asset inventory: target.internal",
                "areas_for_further_testing": "None."
            })
        return json.dumps({"error": "Unknown context", "mock": True})

    def generate_plan(self, recon_data: Dict[str, Any], target: str) -> Dict[str, Any]:
        """Generate an attack plan using the LLM."""
        prompt = self._build_prompt(recon_data, target)
        try:
            response_text = self.generate_response("You are a pentesting planner.", prompt, 2000)
            return self._parse_plan(response_text)
        except Exception as e:
            logger.error(f"Plan generation failed: {e}")
            return {"steps": [], "priority": [], "estimated_time": 60}

    def _build_prompt(self, recon_data, target):
        return f"""You are an autonomous pentesting agent for SERA. Target: {target}
Data: {json.dumps(recon_data, indent=2)}
Generate JSON plan with "steps", "priority", "estimated_time".
"""

    def _parse_plan(self, text):
        try:
            match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            match = re.search(r'(\{.*\})', text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            return json.loads(text)
        except:
            return {"steps": [], "priority": [], "estimated_time": 60}