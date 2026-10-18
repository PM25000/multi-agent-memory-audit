# Third-party notices

## AgentDojo
Source and adapted dummy tasks originate from ethz-spylab/agentdojo, pinned commit
089ed468cf3ed0322acc66b0211f26d9d90dbf60, MIT license.
https://github.com/ethz-spylab/agentdojo/tree/089ed468cf3ed0322acc66b0211f26d9d90dbf60
The upstream license is supplied in LICENSES/AgentDojo-MIT.txt. Author-created
protocols/analyses do not replace the upstream license on original source/data.
S6 includes only source and data needed for the native runtime, with original
file hashes; upstream development documentation/tests/examples are omitted.

## AutoGen Core
The historical native delivery driver uses Microsoft AutoGen Core 0.7.5.
https://github.com/microsoft/autogen (MIT for code; upstream documentation may
have separate terms). No AutoGen library, virtual environment, or upstream
documentation is bundled. Its version/source identities remain in the original
protocol metadata. Offline wire verification reads saved native events and actual
SDK request arguments; it does not execute the message-passing runtime again.

## Tokenizers
Qwen/Qwen3.6-27B @ 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9;
zai-org/GLM-5.2 @ cf457fa734ab149ffef225f80893eb38c6ff5cdc.
Tokenizer binaries/templates are not redistributed. S6 TOKENIZER_ASSETS.json
supplies exact source URLs and hashes for optional native/token replay.
Obtain assets separately under their upstream terms. They are not relicensed
under this repository's MIT or CC BY license.

## Model/provider outputs
Saved experimental requests/responses, usage, identity and failure records are
provided as research evidence. No model weights or third-party service accounts
are included. Model/provider outputs are not asserted to be wholly author-owned
or automatically relicensed by DATA_LICENSE.md. Provider/model provenance is
retained in the records; no additional rights in upstream outputs are granted.

Other Python dependencies are referenced by pinned versions, not bundled.
