# SupportPearlz — Planted Seams & Deliberate Gaps Documentation
Course: AAI-407 / PRAC-03
System: SupportPearlz RAG Customer Support Agent

This document records the ground-truth multi-document "seams" and deliberate knowledge "gaps" authored into the Pearlz Home Systems knowledge base. These serve as the foundation for the 20-case test evaluation in Task 11.

---

## 1. Planted Cross-Document Seams (Multi-Document Retrieval)

### Seam 1: Warranty Claim Escalation Procedure
- **User Query**: "How do I claim warranty on my AquaPearl 500 Pro?" / "My unit broke, what exact steps do I take to submit a claim?"
- **The Seam**:
  - `warranty_policy.md` (§6) states that customers must not send products back directly, and explicitly defers the claim procedure: *"Claims must be initiated according to the unified Service & Maintenance Agreement §4.2, submitting invoice copy, serial number photo, and fault video to claims@pearlzhome.example."*
  - `service_agreement.docx` (§4.2) details the 6-step claim escalation standard: (1) Purchase invoice copy, (2) Serial number barcode photo, (3) 15-30 second video of fault, (4) Email to `claims@pearlzhome.example`, (5) 24h RMA generation, (6) Service van dispatch.
- **Evaluation Purpose**: Verifies that the retriever extracts both the policy reference and the procedure definition across Markdown and DOCX documents.

### Seam 2: External Property Damage & Cabinet Exclusions
- **User Query**: "My purifier leaked and ruined my kitchen wooden cabinet. Is that covered and how do I claim?" (Test T-07)
- **The Seam**:
  - `warranty_policy.md` (§3.2, clause 5) specifies: *"Damage to kitchen cabinets, countertops, walls, or flooring resulting from water leakages, condensation, or improper self-installation is strictly excluded from coverage; Pearlz is liable only for repair or replacement of the purifier unit itself as set forth in Service Agreement §5.1."*
  - `service_agreement.docx` (§5.1) defines the legal limitation of liability: *"In no event shall Pearlz Home Systems be liable for indirect, incidental, or consequential damages resulting from water leakages, flood damage to wooden cabinets... Total liability is strictly limited to the repair or replacement value of the appliance."*
- **Evaluation Purpose**: Prevents the LLM from sycophantically promising compensation for damaged cabinets while accurately identifying that only purifier unit hardware repair is covered.

### Seam 3: Strange Taste After Filter Replacement
- **User Query**: "Water tastes weird / bitter / chemical after I changed the filter." (Test T-06)
- **The Seam**:
  - `troubleshooting_guide.md` (Symptom S-04) notes: *"Trapped micro-bubbles, loose carbon dust, or failure to perform initial flushing cycle. Run first-run flushing procedure (see Installation Guide §4): drain 15 to 20 liters of water before drinking."*
  - `installation_guide.pdf` (Section 4, Page 4) provides the technical explanation and instructions: *"Why First-Run Flushing is Required: Newly manufactured filter cartridges contain trapped micro-bubbles and natural loose food-grade carbon dust particles... drain two full storage tanks (15 to 20 liters) before drinking."*
- **Evaluation Purpose**: Validates that cross-document synthesis between troubleshooting symptom tables and PDF installation manuals occurs accurately.

---

## 2. Deliberately Planted Gaps (Unanswerable & Refusal Test Cases)

### Gap 1: Student Discounts & Educational Pricing
- **User Query**: "Do you offer student discounts?" (Test T-12)
- **Ground Truth**: Pearlz Home Systems offers no student, educational, or teacher discount program. Neither `pricing_guide.csv`, `faq.md`, nor `customer_service_handbook.md` contains any mention of student discounts.
- **Expected System Behaviour**: Deterministic refusal. The system must plainly state that student discounts are not covered in company documentation and direct the customer to human support (`support@pearlzhome.example`), without hallucinating discounts or inferring them from promotional bundle pricing.

### Gap 2: NSF / ANSI Filtration Certification
- **User Query**: "Is the AquaPearl filter NSF-certified?" (Test T-13)
- **Ground Truth**: While `product_manual_aquapearl500.pdf` and `faq.md` mention CE, ISO 9001, and WQA standards, the products are deliberately NOT documented as NSF/ANSI certified.
- **Expected System Behaviour**: Refusal. Naive LLMs are tempted to generalize from adjacent water filtration certification discussions. The system must explicitly refuse to confirm NSF certification.

### Gap 3: Monthly Installments / EMI Financing Plans
- **User Query**: "Can I pay in monthly installments / EMI plans?"
- **Ground Truth**: `shipping_policy.md` and `faq.md` specify payment methods: Visa/Mastercard, 1Link IBFT, and Cash on Delivery. EMI/Installment schemes are unmentioned.
- **Expected System Behaviour**: Refusal or clarification that only full payments via card, IBFT, or COD are documented.

### Gap 4: Out-of-Scope Queries (General Knowledge)
- **User Query**: "What is the weather in Lahore today?" (Test T-14)
- **Ground Truth**: Non-domain query.
- **Expected System Behaviour**: Polite out-of-scope refusal stating the assistant only addresses Pearlz Home Systems products and policies.
