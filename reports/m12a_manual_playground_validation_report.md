# M12A Manual Playground Validation Report

## Purpose

This report manually validates the true free-form retrieval demo using realistic telecom diagram queries. The goal is to check whether M12A retrieves useful diagrams when a user enters natural technical queries directly, without prepared-query mapping.

Scoring:
- Top-1 relevance: 0 = wrong, 1 = partially relevant, 2 = clearly relevant
- Top-5 contains relevant: YES/NO
- Manual quality: Good / Mixed / Poor

## Query-Level Review

| No. | Query | Top-1 relevance | Top-5 contains relevant | Manual quality | Comment |
|---|---|---:|---|---|---|
| 1 | handover failure call flow between UE and eNodeB | 2 | YES | Good | Rank 1 retrieves S1-based handover reject/failure call-flow with UE, Source eNodeB, Target eNodeB, MME and gateway entities. Rank 2 and Rank 4 are also handover reject variants. |
| 2 | LTE user plane protocol stack PDCP RLC MAC PHY | 2 | YES | Good | Rank 1 retrieves a user-plane protocol stack and the top results include PDCP/RLC/MAC/PHY-style protocol stack diagrams. Some results are 5G/L2 relay variants rather than exact LTE-only diagrams, but the core protocol-stack intent is captured well. |
| 3 | RRC connection establishment procedure ladder diagram | 2 | YES | Good | Rank 1 retrieves a connection establishment ladder diagram involving Remote UE, Relay UE, and gNB. Top-5 also contains RRC/context/re-establishment and UE registration sequence diagrams. The result is relevant, although not limited to the classic LTE RRC connection setup flow. |
| 4 | NAS authentication and key agreement procedure | 2 | YES | Good | Rank 1 retrieves a network access authentication procedure, while the top-5 includes EAP-TLS authentication, NAS/security-context related AMF re-allocation, and authentication/authorization procedures. The exact key-agreement-specific result appears slightly lower, but the top-5 is clearly authentication-focused. |
| 5 | AMF SMF UPF PDU session establishment architecture | 2 | YES | Good | Rank 1 retrieves an MBS-related PDU Session Establishment procedure and Rank 2 retrieves a UE-requested PDU Session Establishment flow, making the top results strongly relevant. Some later top-5 results are only partially related through SMF/UPF or authentication terms, but the core PDU-session intent is captured well. |
| 6 | gNB CU DU split F1 interface architecture | 2 | YES | Good | Rank 1 directly retrieves an F1 startup/cell activation procedure involving gNB-DU, gNB-CU, 5GC, and F1 setup messages. Rank 2 is also CU/DU-related, while some top-5 results are broader NRM/management architecture fragments rather than exact F1-interface flows. Overall the CU/DU/F1 intent is captured well, with some mixed results after the top ranks. |
| 7 | random access procedure MSG1 MSG2 MSG3 MSG4 | 0 | PARTIAL | Mixed/Poor | Rank 1 retrieves AMF selection rather than a standalone random access MSG1-MSG4 procedure. Some top-5 diagrams include random-access-related steps inside broader handover or node-change procedures, but the desired MSG1-MSG4 random access flow is not clearly retrieved near the top. This is a useful failure case showing the limitation of caption/context-only BM25 when key terms are mainly inside the diagram. |
| 8 | UE context release request message flow | 1 | PARTIAL | Mixed | Rank 1 retrieves a PDU session modification/leave procedure rather than a direct UE Context Release Request flow. However, ranks 2-4 contain handover or SeNB release message-flow diagrams with UE Context Release steps, so the top-5 includes partially relevant context-release evidence. This shows that message-level queries can be mixed when the key phrase is mainly inside the diagram rather than in the caption. |
| 9 | paging procedure in idle mode | 2 | YES | Good | Rank 1 retrieves a mobile terminating call in idle mode and visibly contains paging request/paging/service request flow. Rank 3 also retrieves a network-triggered service request with paging-related steps, and Rank 5 covers the overall idle-mode process. The result is relevant, although the most explicitly titled CM-IDLE paging-filtering result appears lower at Rank 8. |
| 10 | radio link failure recovery procedure | 2 | YES | Good/Mixed | Rank 1 retrieves an RRC re-establishment procedure, which is closely related to radio link failure recovery. However, several other top-5 results are generic WLAN access or SN modification procedures, while the explicitly titled backhaul-RLF recovery result appears at Rank 10. This is a partial-success case with a strong top result but mixed follow-up ranks. |

## Summary

- Number of queries checked: 10
- Top-1 clearly relevant count: 8/10
- Top-5 relevant count: 8 YES, 2 PARTIAL, 0 NO
- Good cases: 7
- Good/Mixed cases: 1
- Mixed or Mixed/Poor cases: 2
- Poor cases: 0

## Observations

- The free-form retrieval demo works best when the query contains technical terms present in captions or context.
- It is strong for protocol names, architecture names, procedure names, and call-flow terms.
- It can retrieve partially related results when the query requires deeper semantic or visual reasoning beyond captions/context.
- The detailed contact sheet and HTML report make manual verification easier than compact contact sheets.

## Claim Boundary

This manual validation supports the claim that M12A is a useful free-form research demo over the local telecom diagram corpus. It does not claim complete retrieval coverage, global benchmark superiority, or visual reasoning replacement.
