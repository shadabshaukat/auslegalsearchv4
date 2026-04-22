"""
OCI GenAI RAG pipeline for auslegalsearchv3.
- Mirrors RAGPipeline but uses Oracle Cloud GenAI as LLM backend.
- Handles OCI credential and config setup.
- Retrieves relevant documents/chunks from the vector store (optionally).
- Sends context, user question, and options to OCI GenAI service.
- Returns model output (QA answer/summary) and relevant document sources.
- No local model dependency.
"""

import os
from typing import List, Optional, Any, Dict

# Import OCI GenAI SDK
try:
    import oci
except ImportError:
    oci = None  # Guard for environments where oci not installed

class OCIGenAIPipeline:
    def __init__(self, compartment_id: str, model_id: str, oci_config: dict = None, region: str = None):
        """
        Args:
            compartment_id (str): OCI Compartment OCID
            model_id (str): OCI GenAI Model OCID or model name
            oci_config (dict, optional): If provided, use as OCI config dict.
            region (str, optional): OCI region
        """
        self.compartment_id = compartment_id
        self.model_id = model_id
        self.oci_config = oci_config or self._default_oci_config(region=region)
        self.region = self.oci_config.get('region')

        # Extra: Print oci_config, model/category details for deep debug
        print("DEBUG: OCIGenAIPipeline init oci_config:", self.oci_config)
        print("DEBUG: OCIGenAIPipeline init model_id:", model_id)
        self.genai_client = self._build_genai_client()

    def _default_oci_config(self, region=None):
        """
        Loads the default OCI config from ~/.oci/config or env vars, can override region.
        """
        config = {
            "user": os.environ.get("OCI_USER_OCID", ""),
            "key_file": os.environ.get("OCI_KEY_FILE", os.path.expanduser("~/.oci/oci_api_key.pem")),
            "fingerprint": os.environ.get("OCI_KEY_FINGERPRINT", ""),
            "tenancy": os.environ.get("OCI_TENANCY_OCID", ""),
            "region": region or os.environ.get("OCI_REGION", "ap-sydney-1"),
        }
        profile = os.environ.get("OCI_CONFIG_PROFILE", "DEFAULT")
        # If ~/.oci/config present, load
        try:
            file_conf = oci.config.from_file("~/.oci/config", profile_name=profile)
            config.update(file_conf)
        except Exception:
            pass
        # Always enforce the values from environment or provided args:
        config["user"] = os.environ.get("OCI_USER_OCID", config.get("user", ""))
        config["key_file"] = os.environ.get("OCI_KEY_FILE", config.get("key_file", os.path.expanduser("~/.oci/oci_api_key.pem")))
        config["fingerprint"] = os.environ.get("OCI_KEY_FINGERPRINT", config.get("fingerprint", ""))
        config["tenancy"] = os.environ.get("OCI_TENANCY_OCID", config.get("tenancy", ""))
        config["region"] = region or os.environ.get("OCI_REGION", config.get("region", "ap-sydney-1"))
        return config

    def _build_genai_client(self):
        """
        Initializes the OCI Generative AI client per Oracle reference.
        """
        if not oci:
            raise ImportError("oci package not installed. Please pip install oci")
        from oci.generative_ai_inference import GenerativeAiInferenceClient
        return GenerativeAiInferenceClient(self.oci_config)

    def query(
        self,
        question: str,
        context_chunks: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
        chunk_metadata: Optional[List[Dict[str, Any]]] = None,
        custom_prompt: Optional[str] = None,
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = 1024,
        repeat_penalty: float = 1.1,
        chat_history: Optional[List[dict]] = None,
        system_prompt: Optional[str] = None,
        model_info: Optional[dict] = None,
    ) -> dict:
        """
        Calls OCI GenAI with context+question, returns dict with answer, sources, etc.
        model_info: Dict from frontend that includes all model dropdown metadata (operation_types, etc).
        """
        # Prompt composition
        if custom_prompt:
            sys_prompt = custom_prompt.strip()
        else:
            sys_prompt = system_prompt or ("You are a legal assistant. Answer only from the provided context. Cite sources. Be concise.")

        # Format the context for prompt as in original RAG pipeline
        if chunk_metadata is None:
            blocks = context_chunks or []
        else:
            # Rich block: concatenate meta + text for each chunk
            blocks = []
            for text, meta in zip(context_chunks or [], chunk_metadata or []):
                meta_str = "\n".join(f"{k}: {v}" for k, v in (meta or {}).items())
                block = (meta_str + "\n---\n" if meta_str else "") + text
                blocks.append(block)
        formatted_context = "\n\n---\n\n".join(blocks)
        prompt = (
            sys_prompt + "\n\n"
            "Based on the following legal documents/chunks, answer the question or summarize as requested.\n"
            "CONTEXT:\n"
            + formatted_context +
            f"\n\nQUESTION: {question}\nANSWER:"
        )

        # OCI Generative AI Inference Call (detect between generate_text and generate_chat)
        # SDK class imports
        try:
            from oci.generative_ai_inference.models import (
                GenerateTextDetails,
                OnDemandServingMode,
                LlamaLlmInferenceRequest,
                GenerateChatDetails,
                ChatMessage,
                ChatCompletionsOptions,
            )
        except ImportError:
            from oci.generative_ai_inference.models import (
                GenerateTextDetails,
                OnDemandServingMode,
                LlamaLlmInferenceRequest,
            )

        try:
            # Always use Oracle's chat() API for OCI GenAI—maximal compatibility for LLM chat models
            print("DEBUG: Using chat() API (forced) for all OCI GenAI models")
            from oci.generative_ai_inference.models import (
                ChatDetails, GenericChatRequest, Message, TextContent, OnDemandServingMode, BaseChatRequest
            )
            serving_mode = OnDemandServingMode(model_id=self.model_id)
            txt_content = TextContent()
            txt_content.text = prompt
            user_msg = Message()
            user_msg.role = "USER"
            user_msg.content = [txt_content]
            chat_req = GenericChatRequest()
            chat_req.api_format = BaseChatRequest.API_FORMAT_GENERIC
            chat_req.messages = [user_msg]
            chat_req.max_tokens = int(max_tokens)
            chat_req.temperature = float(temperature)
            chat_req.frequency_penalty = 0
            chat_req.presence_penalty = 0
            chat_req.top_p = float(top_p)
            chat_detail = ChatDetails()
            chat_detail.serving_mode = serving_mode
            chat_detail.chat_request = chat_req
            chat_detail.compartment_id = self.compartment_id
            chat_response = self.genai_client.chat(chat_detail)
            answer = getattr(chat_response, "data", None)
            if hasattr(answer, "choices") and answer.choices and hasattr(answer.choices[0], "message"):
                out = answer.choices[0].message
            elif hasattr(answer, "text"):
                out = answer.text
            else:
                out = str(answer)
            answer = out
            return {
                "answer": answer,
                "sources": sources if sources else [],
                "contexts": context_chunks,
                "chunk_metadata": chunk_metadata if chunk_metadata else [],
            }
        except Exception as e:
            return {
                "answer": f"Error querying OCI GenAI: {e}",
                "sources": sources if sources else [],
                "contexts": context_chunks,
                "chunk_metadata": chunk_metadata if chunk_metadata else [],
            }
