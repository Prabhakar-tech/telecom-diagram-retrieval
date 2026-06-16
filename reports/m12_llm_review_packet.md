# M12 Manual LLM Review Packet

This packet contains 10 diverse queries sampled from the 50-query validation set. Use the provided prompt and contact sheet path to manually evaluate the retrieved candidates using an LLM agent (Claude/Gemini/Antigravity).

## Review Instructions
For each query, copy the **Prompt** text, attach the corresponding **Contact Sheet**, and send it to the LLM. Record the LLM's answers in the `reports/m12_50_query_manual_review_template.csv`.

---

### Query 1: q1_1
**Query Text**: handover failure call flow between UE and eNodeB
**Intended Query Type**: Q1 (Direct Caption)
**Contact Sheet Path**: `reports/m12_50_query_contact_sheets/q1_1_contact_sheet.png`

**Prompt**:
> You are judging candidate telecom technical diagrams for a retrieval system. You are not performing full-corpus retrieval. You are only judging the provided candidates. Given the user query "handover failure call flow between UE and eNodeB" and the attached candidate list/contact sheet, mark whether the top-5 contains a relevant diagram, choose the best candidate, and briefly explain why.

---

### Query 2: q1_8
**Query Text**: gNB CU DU split architecture
**Intended Query Type**: Q1 (Direct Caption)
**Contact Sheet Path**: `reports/m12_50_query_contact_sheets/q1_8_contact_sheet.png`

**Prompt**:
> You are judging candidate telecom technical diagrams for a retrieval system. You are not performing full-corpus retrieval. You are only judging the provided candidates. Given the user query "gNB CU DU split architecture" and the attached candidate list/contact sheet, mark whether the top-5 contains a relevant diagram, choose the best candidate, and briefly explain why.

---

### Query 3: q1_12
**Query Text**: LTE protocol stack user plane
**Intended Query Type**: Q1 (Direct Caption)
**Contact Sheet Path**: `reports/m12_50_query_contact_sheets/q1_12_contact_sheet.png`

**Prompt**:
> You are judging candidate telecom technical diagrams for a retrieval system. You are not performing full-corpus retrieval. You are only judging the provided candidates. Given the user query "LTE protocol stack user plane" and the attached candidate list/contact sheet, mark whether the top-5 contains a relevant diagram, choose the best candidate, and briefly explain why.

---

### Query 4: q2_3
**Query Text**: show me the message sequence for setting up an RRC connection
**Intended Query Type**: Q2 (Paraphrased Question)
**Contact Sheet Path**: `reports/m12_50_query_contact_sheets/q2_3_contact_sheet.png`

**Prompt**:
> You are judging candidate telecom technical diagrams for a retrieval system. You are not performing full-corpus retrieval. You are only judging the provided candidates. Given the user query "show me the message sequence for setting up an RRC connection" and the attached candidate list/contact sheet, mark whether the top-5 contains a relevant diagram, choose the best candidate, and briefly explain why.

---

### Query 5: q2_10
**Query Text**: how do AMF and SMF interact with UPF for PDU sessions
**Intended Query Type**: Q2 (Paraphrased Question)
**Contact Sheet Path**: `reports/m12_50_query_contact_sheets/q2_10_contact_sheet.png`

**Prompt**:
> You are judging candidate telecom technical diagrams for a retrieval system. You are not performing full-corpus retrieval. You are only judging the provided candidates. Given the user query "how do AMF and SMF interact with UPF for PDU sessions" and the attached candidate list/contact sheet, mark whether the top-5 contains a relevant diagram, choose the best candidate, and briefly explain why.

---

### Query 6: q2_17
**Query Text**: show the interaction between UE and EPC during attach
**Intended Query Type**: Q2 (Paraphrased Question)
**Contact Sheet Path**: `reports/m12_50_query_contact_sheets/q2_17_contact_sheet.png`

**Prompt**:
> You are judging candidate telecom technical diagrams for a retrieval system. You are not performing full-corpus retrieval. You are only judging the provided candidates. Given the user query "show the interaction between UE and EPC during attach" and the attached candidate list/contact sheet, mark whether the top-5 contains a relevant diagram, choose the best candidate, and briefly explain why.

---

### Query 7: q3_1
**Query Text**: When the UE experiences deteriorating signal conditions from the serving eNodeB, it sends a measurement report indicating that a neighbor cell is stronger. The serving eNodeB makes a handover decision and attempts to prepare the target eNodeB. However, if the target eNodeB cannot admit the UE due to lack of resources, it sends a Handover Preparation Failure message back to the source eNodeB.
**Intended Query Type**: Q3 (Context-extracted Query)
**Contact Sheet Path**: `reports/m12_50_query_contact_sheets/q3_1_contact_sheet.png`

**Prompt**:
> You are judging candidate telecom technical diagrams for a retrieval system. You are not performing full-corpus retrieval. You are only judging the provided candidates. Given the user query "When the UE experiences deteriorating signal conditions from the serving eNodeB..." and the attached candidate list/contact sheet, mark whether the top-5 contains a relevant diagram, choose the best candidate, and briefly explain why.

---

### Query 8: q3_5
**Query Text**: In 5G NR, the gNB can be split into a Central Unit (CU) and one or more Distributed Units (DUs). The CU handles higher layer protocols like RRC and PDCP, while the DU handles lower layers like RLC, MAC, and PHY. They communicate over the F1 interface, allowing for flexible deployment scenarios.
**Intended Query Type**: Q3 (Context-extracted Query)
**Contact Sheet Path**: `reports/m12_50_query_contact_sheets/q3_5_contact_sheet.png`

**Prompt**:
> You are judging candidate telecom technical diagrams for a retrieval system. You are not performing full-corpus retrieval. You are only judging the provided candidates. Given the user query "In 5G NR, the gNB can be split into a Central Unit (CU)..." and the attached candidate list/contact sheet, mark whether the top-5 contains a relevant diagram, choose the best candidate, and briefly explain why.

---

### Query 9: q3_8
**Query Text**: The LTE user plane protocol stack consists of several layers between the UE and the eNodeB. The Packet Data Convergence Protocol (PDCP) handles IP header compression and ciphering. The Radio Link Control (RLC) handles segmentation and reassembly. The Medium Access Control (MAC) handles multiplexing and HARQ. The Physical layer (PHY) handles modulation and coding.
**Intended Query Type**: Q3 (Context-extracted Query)
**Contact Sheet Path**: `reports/m12_50_query_contact_sheets/q3_8_contact_sheet.png`

**Prompt**:
> You are judging candidate telecom technical diagrams for a retrieval system. You are not performing full-corpus retrieval. You are only judging the provided candidates. Given the user query "The LTE user plane protocol stack consists of several layers..." and the attached candidate list/contact sheet, mark whether the top-5 contains a relevant diagram, choose the best candidate, and briefly explain why.

---

### Query 10: q3_11
**Query Text**: During the NAS authentication and key agreement procedure, the MME sends an Authentication Request to the UE containing a random challenge (RAND) and an authentication token (AUTN). The UE verifies the AUTN to authenticate the network. It then computes a response (RES) and sends it back in an Authentication Response message.
**Intended Query Type**: Q3 (Context-extracted Query)
**Contact Sheet Path**: `reports/m12_50_query_contact_sheets/q3_11_contact_sheet.png`

**Prompt**:
> You are judging candidate telecom technical diagrams for a retrieval system. You are not performing full-corpus retrieval. You are only judging the provided candidates. Given the user query "During the NAS authentication and key agreement procedure..." and the attached candidate list/contact sheet, mark whether the top-5 contains a relevant diagram, choose the best candidate, and briefly explain why.
