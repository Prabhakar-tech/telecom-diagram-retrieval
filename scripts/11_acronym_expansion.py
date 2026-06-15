import json
import re
import pandas as pd
from collections import Counter
from pathlib import Path
import random

def extract_acronyms(text):
    if not isinstance(text, str):
        return []
    candidates = []
    words = re.findall(r'\b[A-Za-z0-9\-]+\b', text)
    for w in words:
        if sum(1 for c in w if c.isupper()) >= 2:
            candidates.append(w)
        elif re.match(r'^[eg][A-Z]{1,2}B[s]?$', w, re.IGNORECASE): # eNB, eNBs, gNB, gNodeB
            candidates.append(w)
    return candidates

def get_predefined_lexicon():
    return {
        "UE": {"expansions": ["User Equipment"], "related_terms": ["mobile terminal", "device"], "action": "expand", "reason": "High frequency telecom core term", "evidence_source": "seeded_3gpp_common"},
        "MCPTT": {"expansions": ["Mission Critical Push To Talk"], "related_terms": ["mission critical service", "push to talk", "group communication"], "action": "expand", "reason": "Core acronym", "evidence_source": "seeded_3gpp_common"},
        "MCData": {"expansions": ["Mission Critical Data"], "related_terms": ["mission critical service", "data service"], "action": "expand", "reason": "Core acronym", "evidence_source": "seeded_3gpp_common"},
        "MCVideo": {"expansions": ["Mission Critical Video"], "related_terms": ["mission critical service", "video service"], "action": "expand", "reason": "Core acronym", "evidence_source": "seeded_3gpp_common"},
        "AMF": {"expansions": ["Access and Mobility Management Function"], "related_terms": ["5G core", "registration", "mobility management"], "action": "expand", "reason": "Core 5G term", "evidence_source": "seeded_3gpp_common"},
        "SMF": {"expansions": ["Session Management Function"], "related_terms": ["PDU session", "session management", "5G core"], "action": "expand", "reason": "Core 5G term", "evidence_source": "seeded_3gpp_common"},
        "UPF": {"expansions": ["User Plane Function"], "related_terms": ["user plane", "packet forwarding", "5G core"], "action": "expand", "reason": "Core 5G term", "evidence_source": "seeded_3gpp_common"},
        "MME": {"expansions": ["Mobility Management Entity"], "related_terms": ["EPC", "mobility management", "LTE core"], "action": "expand", "reason": "Core LTE term", "evidence_source": "seeded_3gpp_common"},
        "PDN": {"expansions": ["Packet Data Network"], "related_terms": ["PDN connection", "packet data", "gateway"], "action": "expand", "reason": "Core networking term", "evidence_source": "seeded_3gpp_common"},
        "PDU": {"expansions": ["Protocol Data Unit"], "related_terms": ["PDU session", "data unit", "packet"], "action": "expand", "reason": "Core networking term", "evidence_source": "seeded_3gpp_common"},
        "QoS": {"expansions": ["Quality of Service"], "related_terms": ["priority", "bearer", "traffic handling"], "action": "expand", "reason": "Core networking term", "evidence_source": "seeded_3gpp_common"},
        "ProSe": {"expansions": ["Proximity Services"], "related_terms": ["device to device", "direct communication", "public safety"], "action": "expand", "reason": "Core acronym", "evidence_source": "seeded_3gpp_common"},
        "AF": {"expansions": ["Application Function"], "related_terms": ["policy authorization", "application server"], "action": "expand", "reason": "Core networking term", "evidence_source": "seeded_3gpp_common"},
        "NPN": {"expansions": ["Non Public Network"], "related_terms": ["private network", "SNPN", "PNI-NPN"], "action": "expand", "reason": "Core acronym", "evidence_source": "seeded_3gpp_common"},
        "HNB": {"expansions": ["Home NodeB"], "related_terms": ["femtocell", "home base station"], "action": "expand", "reason": "Core access term", "evidence_source": "seeded_3gpp_common"},
        "HNBAP": {"expansions": ["Home NodeB Application Part"], "related_terms": ["HNB signalling", "Iuh interface"], "action": "expand", "reason": "Core protocol term", "evidence_source": "seeded_3gpp_common"},
        "RANAP": {"expansions": ["Radio Access Network Application Part"], "related_terms": ["Iu interface", "RAN signalling"], "action": "expand", "reason": "Core protocol term", "evidence_source": "seeded_3gpp_common"},
        "RRC": {"expansions": ["Radio Resource Control"], "related_terms": ["connection setup", "connection release", "radio signalling"], "action": "expand", "reason": "Core protocol term", "evidence_source": "seeded_3gpp_common"},
        "NAS": {"expansions": ["Non Access Stratum"], "related_terms": ["registration", "mobility management", "session management"], "action": "expand", "reason": "Core protocol term", "evidence_source": "seeded_3gpp_common"},
        "gNB": {"expansions": ["next generation NodeB"], "related_terms": ["5G base station", "NG-RAN"], "action": "expand", "reason": "Core access term", "evidence_source": "seeded_3gpp_common"},
        "eNB": {"expansions": ["E-UTRAN NodeB"], "related_terms": ["LTE base station", "E-UTRAN"], "action": "expand", "reason": "Core access term", "evidence_source": "seeded_3gpp_common"},
        "NG-RAN": {"expansions": ["Next Generation Radio Access Network"], "related_terms": ["5G radio access network", "gNB"], "action": "expand", "reason": "Core access term", "evidence_source": "seeded_3gpp_common"},
        "E-UTRAN": {"expansions": ["Evolved Universal Terrestrial Radio Access Network"], "related_terms": ["LTE radio access network", "eNB"], "action": "expand", "reason": "Core access term", "evidence_source": "seeded_3gpp_common"},
        "UTRAN": {"expansions": ["Universal Terrestrial Radio Access Network"], "related_terms": ["3G radio access network", "NodeB"], "action": "expand", "reason": "Core access term", "evidence_source": "seeded_3gpp_common"},
        "CN": {"expansions": ["Core Network"], "related_terms": ["network core", "EPC", "5GC"], "action": "expand", "reason": "Core networking term", "evidence_source": "seeded_3gpp_common"},
        "EPC": {"expansions": ["Evolved Packet Core"], "related_terms": ["LTE core network", "MME", "SGW", "PGW"], "action": "expand", "reason": "Core networking term", "evidence_source": "seeded_3gpp_common"},
        "5GC": {"expansions": ["5G Core"], "related_terms": ["AMF", "SMF", "UPF"], "action": "expand", "reason": "Core networking term", "evidence_source": "seeded_3gpp_common"},
        "SGW": {"expansions": ["Serving Gateway"], "related_terms": ["EPC gateway", "user plane"], "action": "expand", "reason": "Core networking term", "evidence_source": "seeded_3gpp_common"},
        "PGW": {"expansions": ["Packet Data Network Gateway"], "related_terms": ["PDN gateway", "EPC gateway"], "action": "expand", "reason": "Core networking term", "evidence_source": "seeded_3gpp_common"},
        "PLMN": {"expansions": ["Public Land Mobile Network"], "related_terms": ["mobile network", "operator network"], "action": "expand", "reason": "Core networking term", "evidence_source": "seeded_3gpp_common"},
        "CSG": {"expansions": ["Closed Subscriber Group"], "related_terms": ["home nodeb access", "subscriber group"], "action": "expand", "reason": "Core concept", "evidence_source": "seeded_3gpp_common"},
        "IMS": {"expansions": ["IP Multimedia Subsystem"], "related_terms": ["SIP", "multimedia service"], "action": "expand", "reason": "Core subsystem", "evidence_source": "seeded_3gpp_common"},
        "RAN": {"expansions": ["Radio Access Network"], "related_terms": ["radio access", "base station"], "action": "expand", "reason": "Core access term", "evidence_source": "seeded_3gpp_common"},
        "UDM": {"expansions": ["Unified Data Management"], "related_terms": ["subscriber data", "5G core"], "action": "expand", "reason": "Core 5G term", "evidence_source": "seeded_3gpp_common"},
        "PCF": {"expansions": ["Policy Control Function"], "related_terms": ["policy control", "5G core"], "action": "expand", "reason": "Core 5G term", "evidence_source": "seeded_3gpp_common"},
        "NEF": {"expansions": ["Network Exposure Function"], "related_terms": ["service exposure", "5G core"], "action": "expand", "reason": "Core 5G term", "evidence_source": "seeded_3gpp_common"},
        "NWDAF": {"expansions": ["Network Data Analytics Function"], "related_terms": ["analytics", "5G core"], "action": "expand", "reason": "Core 5G term", "evidence_source": "seeded_3gpp_common"},
        "HSS": {"expansions": ["Home Subscriber Server"], "related_terms": ["subscriber database", "EPC"], "action": "expand", "reason": "Core LTE term", "evidence_source": "seeded_3gpp_common"},
        "SGSN": {"expansions": ["Serving GPRS Support Node"], "related_terms": ["packet core", "mobility management"], "action": "expand", "reason": "Core 3G/LTE term", "evidence_source": "seeded_3gpp_common"},
        "EPS": {"expansions": ["Evolved Packet System"], "related_terms": ["LTE system", "EPC"], "action": "expand", "reason": "Core LTE term", "evidence_source": "seeded_3gpp_common"},
        "MBMS": {"expansions": ["Multimedia Broadcast Multicast Service"], "related_terms": ["broadcast service", "multicast service"], "action": "expand", "reason": "Core broadcasting term", "evidence_source": "seeded_3gpp_common"},
        "V2X": {"expansions": ["Vehicle to Everything"], "related_terms": ["vehicle communication", "sidelink"], "action": "expand", "reason": "Core V2X term", "evidence_source": "seeded_3gpp_common"},
        "PC5": {"expansions": ["PC5 interface"], "related_terms": ["sidelink", "direct communication"], "action": "expand", "reason": "Core V2X term", "evidence_source": "seeded_3gpp_common"},
        "SIP": {"expansions": ["Session Initiation Protocol"], "related_terms": ["IMS signaling", "multimedia session"], "action": "expand", "reason": "Core IMS term", "evidence_source": "seeded_3gpp_common"},
        "IWF": {"expansions": ["Interworking Function"], "related_terms": ["interworking", "network interworking"], "action": "expand", "reason": "Core interworking term", "evidence_source": "seeded_3gpp_common"},

        "NOTE": {"expansions": [], "related_terms": [], "action": "skip", "reason": "section marker/noise", "evidence_source": "skip_ambiguous"},
        "ID": {"expansions": [], "related_terms": [], "action": "skip", "reason": "generic identifier term", "evidence_source": "skip_ambiguous"},
        "TS": {"expansions": [], "related_terms": [], "action": "skip", "reason": "can mean Technical Specification but too generic and frequent", "evidence_source": "skip_ambiguous"},
        "MC": {"expansions": [], "related_terms": [], "action": "skip", "reason": "ambiguous abbreviation; use MCPTT/MCData/MCVideo instead", "evidence_source": "skip_ambiguous"},
        "IP": {"expansions": [], "related_terms": [], "action": "skip", "reason": "too generic; can cause broad false matches", "evidence_source": "skip_ambiguous"},
        "3GPP": {"expansions": [], "related_terms": [], "action": "skip", "reason": "too generic and frequent", "evidence_source": "skip_ambiguous"},
        "API": {"expansions": [], "related_terms": [], "action": "skip", "reason": "too generic and frequent", "evidence_source": "skip_ambiguous"},
        "DNS": {"expansions": [], "related_terms": [], "action": "skip", "reason": "too generic and frequent", "evidence_source": "skip_ambiguous"},
        "HTTP": {"expansions": [], "related_terms": [], "action": "skip", "reason": "too generic and frequent", "evidence_source": "skip_ambiguous"},

        "SN": {"expansions": [], "related_terms": [], "action": "needs_review", "reason": "can mean sequence number, serving network, serial number depending context", "evidence_source": "manual_review_needed"},
        "GW": {"expansions": [], "related_terms": [], "action": "needs_review", "reason": "gateway is useful but broad; prefer SGW/PGW explicitly where possible", "evidence_source": "manual_review_needed"},
        "AS": {"expansions": [], "related_terms": [], "action": "needs_review", "reason": "unclear or low-confidence candidate", "evidence_source": "manual_review_needed"},
        "MN": {"expansions": [], "related_terms": [], "action": "needs_review", "reason": "unclear or low-confidence candidate", "evidence_source": "manual_review_needed"}
    }

def get_aliases():
    return {
        "UEs": "UE",
        "HNBs": "HNB",
        "eNodeB": "eNB",
        "eNodeBs": "eNB",
        "eNBs": "eNB",
        "gNodeB": "gNB",
        "gNodeBs": "gNB",
        "gNBs": "gNB",
        "RRCs": "RRC",
        "PDUs": "PDU",
        "NPNs": "NPN"
    }

def process_lexicon():
    print("Enriching lexicon...")
    root = Path("/DATA5/prabhakar/telecom_retrieval")
    with open(root / "reports/m65_acronym_candidates.json") as f:
        candidates = json.load(f)
        
    known_lexicon = get_predefined_lexicon()
    aliases = get_aliases()
    
    enriched_lexicon = {}
    
    # Prepopulate with all known lexicon items
    for acr, info in known_lexicon.items():
        enriched_lexicon[acr] = info.copy()
        enriched_lexicon[acr]["acronym"] = acr
        enriched_lexicon[acr]["aliases"] = [k for k, v in aliases.items() if v == acr]
    
    # Process candidates
    for acr in candidates.keys():
        if acr not in enriched_lexicon and acr not in aliases:
            enriched_lexicon[acr] = {
                "acronym": acr,
                "aliases": [],
                "expansions": [],
                "related_terms": [],
                "action": "needs_review",
                "reason": "unclear or low-confidence candidate",
                "evidence_source": "manual_review_needed"
            }
            
    with open(root / "reports/m65_acronym_lexicon.json", "w") as f:
        json.dump(enriched_lexicon, f, indent=2)
        
    return enriched_lexicon

def get_phrase_expansions(text):
    phrase_matches = []
    
    # SGW Phrase Rule
    if "Serving GW" in text:
        phrase_matches.append({
            "matched": "Serving GW",
            "expansions": ["Serving Gateway"],
            "related_terms": ["EPC gateway", "user plane"]
        })
        
    # PGW Phrase Rule
    if "PDN GW" in text or "Packet Data Network GW" in text:
        phrase_matches.append({
            "matched": "PDN GW" if "PDN GW" in text else "Packet Data Network GW",
            "expansions": ["Packet Data Network Gateway"],
            "related_terms": ["PDN gateway", "EPC gateway"]
        })
        
    return phrase_matches

def expand_query(text, lexicon, aliases):
    words = extract_acronyms(text)
    
    matched = []
    skipped = []
    added_terms = []
    token_count = 0
    
    # Handle phrases first
    phrase_matches = get_phrase_expansions(text)
    for p in phrase_matches:
        matched.append(p["matched"])
        if p["expansions"]:
            exp = p["expansions"][0]
            added_terms.append(exp)
            token_count += len(exp.split())
        rel_count = 0
        for rel in p["related_terms"]:
            if rel_count >= 2: break
            if token_count + len(rel.split()) > 20: break
            added_terms.append(rel)
            token_count += len(rel.split())
            rel_count += 1
            
    seen_words = set(matched) # avoid duplicate expansions

    # Handle acronym tokens
    for raw_w in set(words):
        # Resolve alias
        w = aliases.get(raw_w, raw_w)
        
        if w in seen_words:
            continue
            
        if w in lexicon:
            if lexicon[w]["action"] == "expand":
                seen_words.add(w)
                matched.append(raw_w if raw_w != w else w)
                
                if lexicon[w]["expansions"]:
                    exp = lexicon[w]["expansions"][0]
                    added_terms.append(exp)
                    token_count += len(exp.split())
                    
                rel_count = 0
                for rel in lexicon[w].get("related_terms", []):
                    if rel_count >= 2: break
                    if token_count + len(rel.split()) > 20: break
                    added_terms.append(rel)
                    token_count += len(rel.split())
                    rel_count += 1
            else:
                skipped.append(raw_w)
                
    expanded_text = f"{text} {' '.join(added_terms)}" if added_terms else text
    return expanded_text, matched, added_terms, skipped

def sample_query_expansions(lexicon):
    print("Generating enriched query expansion samples...")
    root = Path("/DATA5/prabhakar/telecom_retrieval")
    
    q1_queries = json.load(open(root / "queries/q1_captions.json"))["queries"]
    q2_queries = json.load(open(root / "queries/q2_paraphrased.json"))["queries"]
    q3_queries = json.load(open(root / "queries/q3_context.json"))["queries"]
    
    all_queries = []
    for q_list, q_type in [(q1_queries, "q1"), (q2_queries, "q2"), (q3_queries, "q3")]:
        random.seed(42)
        random.shuffle(q_list)
        all_queries.extend([(q, q_type) for q in q_list[:100]]) # pull from a larger pool
        
    # specifically find the user's requested test cases if they aren't hit randomly
    test_queries = [
        "This procedure is used to hand over a UE from a source eNodeB to a target eNodeB using X2 when the MME is unchanged and decides that the Serving GW is also unchanged.",
        "What diagram shows the PDN GW allocating an IP address?",
        "UEs in the network connect to the core."
    ]
    for i, tq in enumerate(test_queries):
        all_queries.insert(i, ({"query_id": f"test_{i}", "text": tq}, "test"))
        
    samples = []
    md_content = ["# Query Expansion Examples\n"]
    
    q1_count, q2_count, q3_count, test_count = 0, 0, 0, 0
    aliases = get_aliases()
    
    for q_data, q_type in all_queries:
        if q_type == "q1" and q1_count >= 10: continue
        if q_type == "q2" and q2_count >= 10: continue
        if q_type == "q3" and q3_count >= 10: continue
        if q_type == "test" and test_count >= 3: continue
        
        q = q_data
        text = q["text"]
        
        expanded_text, matched, added_terms, skipped = expand_query(text, lexicon, aliases)
        
        if added_terms or skipped or q_type == "test":
            if q_type == "q1": q1_count += 1
            if q_type == "q2": q2_count += 1
            if q_type == "q3": q3_count += 1
            if q_type == "test": test_count += 1
            
            result = {
                "query_id": q["query_id"],
                "original_query": text,
                "expanded_query": expanded_text,
                "matched_acronyms": matched,
                "added_terms": added_terms,
                "skipped_acronyms": skipped
            }
            samples.append(result)
            
            md_content.append(f"### Query: `{q['query_id']}`")
            md_content.append(f"**Original**: {text}")
            md_content.append(f"**Expanded**: {expanded_text}")
            md_content.append(f"**Matched Acronyms**: {', '.join(matched) if matched else 'None'}")
            md_content.append(f"**Added Terms**: {', '.join(added_terms) if added_terms else 'None'}")
            md_content.append(f"**Skipped Acronyms**: {', '.join(skipped) if skipped else 'None'}\n")
            
    with open(root / "reports/m65_query_expansion_examples.json", "w") as f:
        json.dump(samples, f, indent=2)
        
    with open(root / "reports/m65_query_expansion_examples.md", "w") as f:
        f.write("\n".join(md_content))
        
    return samples

if __name__ == "__main__":
    lexicon = process_lexicon()
    sample_query_expansions(lexicon)
