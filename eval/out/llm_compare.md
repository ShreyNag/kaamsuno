# Interpreter LLM comparison

| model | set | n | next action ok | field checks ok | calls | interp p50 ms | interp p95 ms | errors |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| openai/gpt-oss-20b | DIA | 18 | 16/18 | 6/8 | 19 | 843 | 1834 | 0 |
| openai/gpt-oss-20b | holdout | 50 | 44/50 | 18/48 | 50 | 848 | 1393 | 0 |
| openai/gpt-oss-20b | all | 68 | 60/68 | 24/56 | 69 | 843 | 1415 | 0 |

Turn-taking items (tail classifier, no LLM): DIA010 reply (other), DIA011 reply (other)

Items where a model missed the action or a field check (✗ = failed checks):

| item | openai/gpt-oss-20b | gold |
|---|---|---|
| DIA010 Umm... matlab... aur ek issue hai. | reply (other) | keep_listening |
| DIA011 Bas... ek aur cheez. | reply (other) | keep_listening |
| DIA013 Nahi, Bay 2 nahi, Bay 3. | reask | reread_location |
| DIA014 Wire se sparks aa rahe hain aur log paas hain | ask_location ✗urgency | emergency |
| DIA020 Cement kam hai, around twenty bags. | ask_location ✗quantity | ask_location |
| SCN0036 pump P07 Bay 3 par has a crack. | ask_urgency ✗incident_type | ask_urgency |
| SCN0066 pump P07 ke paas problem hai, has a crack; ba | ask_urgency ✗incident_type | ask_urgency |
| SCN0096 Sir, pump P07 has a crack, location mixing ar | ask_urgency ✗incident_type | ask_urgency |
| SCN0111 Sir, pump P07 has stopped working, location l | ask_location ✗location | ask_urgency |
| SCN0126 pump P07 floor 5 par has a crack. | ask_incident_type ✗incident_type | ask_urgency |
| SCN0156 Bay 3 mein generator G03 ka issue hai—has a c | ask_incident_type ✗incident_type | ask_urgency |
| SCN0186 generator G03 basement level par has a crack. | ask_urgency ✗incident_type | ask_urgency |
| SCN0216 generator G03 ke paas problem hai, has a crac | ask_urgency ✗location,incident_type | ask_urgency |
| SCN0231 generator G03 ke paas problem hai, has stoppe | ask_urgency ✗location | ask_urgency |
| SCN0246 floor 5 mein generator G03 ka issue hai—has a | ask_urgency ✗incident_type | ask_urgency |
| SCN0276 conveyor C12 Bay 3 par has a crack. | ask_urgency ✗incident_type | ask_urgency |
| SCN0306 conveyor C12 ke paas problem hai, has a crack | ask_urgency ✗location,incident_type | ask_urgency |
| SCN0336 Sir, conveyor C12 has a crack, location mixin | ask_urgency ✗incident_type | ask_urgency |
| SCN0351 Sir, conveyor C12 has stopped working, locati | ask_location ✗location | ask_urgency |
| SCN0366 conveyor C12 floor 5 par has a crack. | ask_incident_type ✗incident_type | ask_urgency |
| SCN0396 Bay 3 mein crane C12 ka issue hai—has a crack | ask_urgency ✗incident_type | ask_urgency |
| SCN0426 crane C12 basement level par has a crack. | ask_urgency ✗incident_type | ask_urgency |
| SCN0456 crane C12 ke paas problem hai, has a crack; m | ask_urgency ✗location,incident_type | ask_urgency |
| SCN0471 crane C12 ke paas problem hai, has stopped wo | ask_urgency ✗location | ask_urgency |
| SCN0486 floor 5 mein crane C12 ka issue hai—has a cra | ask_urgency ✗incident_type | ask_urgency |
| SCN0501 Sir, panel B15 has stopped working, location  | ask_urgency ✗asset_or_equipment | ask_urgency |
| SCN0516 Bay 3 mein panel B15 ka issue hai—has a crack | ask_incident_type ✗asset_or_equipment,incident_type | ask_urgency |
| SCN0546 panel B15 basement level par has a crack. | ask_urgency ✗incident_type | ask_urgency |
| SCN0561 panel B15 ke paas problem hai, has stopped wo | ask_urgency ✗location | ask_urgency |
| SCN0576 panel B15 ke paas problem hai, has a crack; m | ask_urgency ✗location,asset_or_equipment,incident_type | ask_urgency |
| SCN0591 panel B15 ke paas problem hai, has stopped wo | ask_urgency ✗location | ask_urgency |
| SCN0606 floor 5 mein panel B15 ka issue hai—has a cra | ask_urgency ✗incident_type | ask_urgency |
| SCN0636 Sir, compressor A10 has a crack, location Bay | ask_urgency ✗incident_type | ask_urgency |
| SCN0666 basement level mein compressor A10 ka issue h | ask_urgency ✗incident_type | ask_urgency |
| SCN0696 compressor A10 mixing area par has a crack. | ask_urgency ✗incident_type | ask_urgency |
