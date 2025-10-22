import time
import xml.etree.ElementTree as ET
import re
from datetime import datetime, timezone
from typing import Any

from openai import OpenAI

from textarena.core import Agent
from mindgames.db.models import Agent as DBAgent
from mindgames.constants import DEFAULT_API_KEY, DEFAULT_BASE_URL

STANDARD_GAME_PROMPT = "You are a competitive game player. Make sure you read the game instructions carefully, and always follow the required format."


class ThinkingAgent(Agent):
    """
    Agent that separates internal reasoning from public communication using XML tags.

    Uses <outloud> XML tag for public-facing actions while allowing internal deliberation.
    Combines strategy prompt with XML formatting instructions.
    """

    XML_INSTRUCTIONS: str = """
## XML instructions

When responding, structure your output as follows:
1. Begin with your internal thinking process (not enclosed in any tags)
2. End with your public action wrapped in <outloud></outloud> tags

The content inside <outloud> tags will be your public action that other players see.
Everything outside the tags is your private thinking that others cannot see.

### Example format

```text
I need to analyze the situation carefully. Based on the current state...
[internal reasoning continues...]
<outloud>I think that Player A has been suspicious because they said ... I vote to eliminate Player B</outloud>
```

### Example voting

Ensure that if you vote (e.g. "[Player B]") that this is outloud!

<outloud>I vote [Player B]</outloud>
"""

    def __init__(
        self,
        agent_db: DBAgent | None = None,
        model_name: str | None = None,
        system_prompt: str = STANDARD_GAME_PROMPT,
    ):
        """
        Initialize the Thinking agent.

        Can be initialized with either a database Agent object or directly with model_name.

        Args:
            agent_db (Optional[DBAgent]): Database Agent object with loaded relationships
            model_name (Optional[str]): Direct model name for simple initialization
            system_prompt (str): The strategy prompt to use
        """
        if agent_db is not None:
            # Initialize from database agent
            self.agent_db = agent_db
            self.model_name = agent_db.model.litellm_model_name

            # Build client configuration based on model settings
            # Default to LiteLLM proxy
            self.client = OpenAI(
                api_key=DEFAULT_API_KEY, base_url=DEFAULT_BASE_URL
            )
        elif model_name is not None:
            # Direct initialization with model name
            self.agent_db = None
            self.model_name = model_name
            # Default to LiteLLM proxy
            self.client = OpenAI(
                api_key=DEFAULT_API_KEY, base_url=DEFAULT_BASE_URL
            )
        else:
            raise ValueError("Either agent_db or model_name must be provided")

        self.full_system_prompt = f"{system_prompt}\n\n{self.XML_INSTRUCTIONS}"
        self.last_llm_metadata = {}
        super().__init__()

    def _parse_xml_response(self, response: str) -> tuple[str, str, bool]:
        """
        Parse XML response to extract public action from <outloud> tags.

        Args:
            response (str): The raw response from the LLM.

        Returns:
            Tuple[str, str, bool]: (full_response, parsed_action, parse_success)
                - full_response: The complete raw response including internal thoughts
                - parsed_action: Only the content from <outloud> tags, or fallback response
                - parse_success: True if <outloud> tags were found, False otherwise
        """
        self.last_raw_response = response

        try:
            # Try to find <outloud> tags using regex first (more robust than XML parsing)
            outloud_pattern = r"<outloud>(.*?)</outloud>"
            matches = re.findall(outloud_pattern, response, re.DOTALL | re.IGNORECASE)

            if matches:
                # Use the last match if there are multiple <outloud> tags
                parsed_action = matches[-1].strip()
                self.last_parsed_action = parsed_action
                return response, parsed_action, True

            # If no <outloud> tags found, try XML parsing as fallback
            # Wrap response in root element for parsing
            xml_wrapped = f"<root>{response}</root>"
            root = ET.fromstring(xml_wrapped)
            outloud_elements = root.findall(".//outloud")

            if outloud_elements:
                # Use the last outloud element if multiple exist
                parsed_action = outloud_elements[-1].text or ""
                self.last_parsed_action = parsed_action.strip()
                return response, parsed_action.strip(), True

        except ET.ParseError as e:
            print(f"XML parsing failed: {e}")

            print("=" * 20)
            print(response)
            print("=" * 20)
        except Exception as e:
            print(f"Unexpected error in XML parsing: {e}")

        # Parse failed - return the response as-is but flag it as failed
        self.last_parsed_action = response.strip()
        return response, response.strip(), False

    def __call__(
        self, observation: str, return_metadata: bool = False
    ) -> str | tuple[str, dict[str, Any]]:
        """
        Process the observation and return only the parsed public action.

        Args:
            observation (str): The input string to process.
            return_metadata (bool): If True, return metadata about the LLM call

        Returns:
            str: The parsed public action from <outloud> tags.
            OR if return_metadata=True:
            Tuple[str, Dict]: (parsed_action, metadata_dict)
        """
        if not isinstance(observation, str):
            raise ValueError(
                f"Observation must be a string. Received type: {type(observation)}"
            )

        # Accumulate all LLM calls made during this agent invocation
        all_llm_calls = []

        # Get the full response from the base class
        full_response, llm_metadata = self._retry_request(
            observation, return_metadata=True
        )
        all_llm_calls.append(llm_metadata["llm_call"])

        # Parse the response to extract public action
        _, parsed_action, parse_success = self._parse_xml_response(full_response)

        # If parsing failed, retry up to 5 times with formatting correction
        max_retries = 5
        retry_count = 0
        last_response = full_response

        while not parse_success and retry_count < max_retries:
            retry_count += 1
            print(f"⚠️  XML parsing failed, retry {retry_count}/{max_retries}...")

            correction_prompt = f"""Your previous response didn't use the required XML format. Please reformat your response to include your action wrapped in <outloud></outloud> tags.

Your previous response was:
{last_response}

Please provide the same action but properly formatted with <outloud> tags."""

            # Make retry call
            retry_response, retry_metadata = self._retry_request(
                correction_prompt, return_metadata=True
            )
            all_llm_calls.append(retry_metadata["llm_call"])

            # Parse the retry response
            _, parsed_action, parse_success = self._parse_xml_response(retry_response)

            # Update last_response for next iteration if needed
            last_response = retry_response

            if parse_success:
                print(f"✅ Retry {retry_count} successful - XML parsed correctly")
            elif retry_count >= max_retries:
                print(
                    f"⚠️  All {max_retries} retries failed - using raw response as fallback"
                )

        # Build combined metadata with all calls
        combined_metadata = {
            "tokens_used": sum(
                call.get("response_tokens", 0) for call in all_llm_calls
            ),
            "llm_calls": all_llm_calls,  # List of all LLM calls
        }

        if return_metadata:
            return parsed_action, combined_metadata
        return parsed_action

    def get_last_raw_response(self) -> str:
        """
        Get the last complete raw response including internal thoughts.

        Returns:
            str: The full raw response from the last call.
        """
        return self.last_raw_response

    def get_last_parsed_action(self) -> str:
        """
        Get the last parsed public action.

        Returns:
            str: The parsed action from the last call.
        """
        return self.last_parsed_action

    def _retry_request(
        self,
        observation: str,
        retries: int = 3,
        delay: int = 5,
        return_metadata: bool = False,
    ) -> str | tuple[str, dict[str, Any]]:
        """
        Attempt to make an API request with retries.

        Args:
            observation (str): The input to process.
            retries (int): The number of attempts to try.
            delay (int): Seconds to wait between attempts.
            return_metadata (bool): If True, return metadata about the LLM call

        Returns:
            str: The response text OR Tuple[str, Dict] if return_metadata=True

        Raises:
            Exception: The last exception caught if all retries fail.
        """
        last_exception = None
        for attempt in range(1, retries + 1):
            try:
                if return_metadata:
                    response, metadata = self._make_request(
                        observation, return_metadata=True
                    )
                    # print(f"\nObservation: {observation}\nResponse: {response}")
                    return response, metadata
                else:
                    response = self._make_request(observation)
                    # print(f"\nObservation: {observation}\nResponse: {response}")
                    return response
            except Exception as e:
                last_exception = e
                # print(f"Attempt {attempt} failed with error: {e}")

                # TODO: replace this to use the tenacity library
                if attempt < retries:
                    time.sleep(delay)
        if last_exception is not None:
            raise last_exception
        else:
            raise Exception("Request failed after all retries")

    def _make_request(
        self, observation: str, return_metadata: bool = False
    ) -> str | tuple[str, dict[str, Any]]:
        """
        Make a single API request to OpenAI and return the generated message.

        Args:
            observation (str): The input string to process.
            return_metadata (bool): If True, return metadata about the LLM call

        Returns:
            str: The generated response text OR Tuple[str, Dict] if return_metadata=True
        """
        messages = [
            {"role": "system", "content": self.full_system_prompt},
            {"role": "user", "content": observation},
        ]

        # Track timing
        request_sent_at = datetime.now(timezone.utc)
        start_time = time.time()

        # Make the API call using the provided model and messages.
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            n=1,
            stop=None,
            max_tokens=8192,
        )

        # Calculate timing
        response_received_at = datetime.now(timezone.utc)
        total_response_time_ms = int((time.time() - start_time) * 1000)

        response_content = completion.choices[0].message.content
        response_text = response_content.strip() if response_content else ""

        if return_metadata:
            # Extract token counts and other metadata
            metadata = {
                "tokens_used": completion.usage.completion_tokens
                if completion.usage
                else None,
                "llm_call": {
                    "messages": messages,
                    "response": completion.model_dump()
                    if hasattr(completion, "model_dump")
                    else dict(completion),
                    "request_sent_at": request_sent_at,
                    "response_received_at": response_received_at,
                    "total_response_time_ms": total_response_time_ms,
                    "response_tokens": completion.usage.completion_tokens
                    if completion.usage
                    else 0,
                    "total_tokens_consumed": completion.usage.total_tokens
                    if completion.usage
                    else 0,
                },
            }
            return response_text, metadata

        return response_text
