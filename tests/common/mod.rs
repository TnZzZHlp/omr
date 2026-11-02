//! Snapshot of a reading, as stored in `tests/*/expected/*.json`.
//!
//! The snapshot keeps what a regression must not change: which option was read,
//! which alternatives were present, the review state, and the candidate and page
//! number. Confidence is stored rounded, so a harmless change in floating point
//! rounding does not fail a test while a real change in separation still does.

use omr::Recognition;
use serde::{Deserialize, Serialize};

/// Confidence is stored to two decimals.
fn bucket(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Snapshot {
    pub page: usize,
    pub template_id: String,
    pub registration: RegistrationSnapshot,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub page_number: Option<Choice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate_number: Option<Candidate>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub barcode: Option<Barcode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub absent: Option<bool>,
    pub selections: Vec<Choice>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub multiple: Vec<MultipleChoice>,
    pub review_count: usize,
    /// Set when the sheet could not be registered at all.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub registration_failed: Option<String>,
    /// Fingerprint of the annotation drawn over the scan.
    pub overlay: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RegistrationSnapshot {
    pub mode: String,
    /// Corner error in scan pixels, rounded to a tenth.
    pub max_residual: f64,
    pub rotation_deg: f64,
    pub scale_x: f64,
    pub scale_y: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Choice {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub question: Option<u32>,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub also_marked: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    pub needs_review: bool,
    pub confidence: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MultipleChoice {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub question: Option<u32>,
    /// Options marked, in declaration order.
    pub selected: Vec<String>,
    pub needs_review: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Candidate {
    pub value: String,
    /// One digit per column; `.` marks a column read as empty.
    pub digits: Vec<String>,
    pub needs_review: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Barcode {
    pub text: String,
    pub check_ok: bool,
}

impl Snapshot {
    /// Builds a snapshot including the annotation fingerprint, as stored.
    pub fn from_parts(recognition: &Recognition, overlay: String) -> Self {
        let registration = &recognition.registration;
        Self {
            page: recognition.page,
            template_id: recognition.template_id.clone(),
            registration: RegistrationSnapshot {
                mode: format!("{:?}", registration.mode),
                max_residual: (registration.max_residual * 10.0).round() / 10.0,
                rotation_deg: (registration.rotation_deg * 100.0).round() / 100.0,
                scale_x: (registration.scale_x * 1000.0).round() / 1000.0,
                scale_y: (registration.scale_y * 1000.0).round() / 1000.0,
            },
            page_number: recognition.page_number.as_ref().map(|number| Choice {
                question: None,
                name: "PNO".to_owned(),
                selected: number.selected.clone(),
                also_marked: Vec::new(),
                reason: number.reason.map(|r| format!("{r:?}")),
                needs_review: number.needs_review,
                confidence: bucket(number.confidence),
            }),
            candidate_number: recognition
                .candidate_number
                .as_ref()
                .map(|candidate| Candidate {
                    value: candidate.value.clone(),
                    digits: candidate
                        .columns
                        .iter()
                        .map(|column| column.selected.clone().unwrap_or_else(|| ".".to_owned()))
                        .collect(),
                    needs_review: candidate.needs_review,
                }),
            barcode: recognition.barcode.as_ref().map(|barcode| Barcode {
                text: barcode.text.clone(),
                check_ok: barcode.check_ok,
            }),
            absent: recognition.absent.as_ref().map(|absent| absent.marked),
            selections: recognition
                .selections
                .iter()
                .map(|selection| Choice {
                    question: selection.question,
                    name: selection.name.clone(),
                    selected: selection.selected.clone(),
                    also_marked: selection.also_marked.clone(),
                    reason: selection.reason.map(|r| format!("{r:?}")),
                    needs_review: selection.needs_review,
                    confidence: bucket(selection.confidence),
                })
                .collect(),
            multiple: recognition
                .multiple
                .iter()
                .map(|reading| MultipleChoice {
                    name: reading.name.clone(),
                    question: reading.question,
                    selected: reading.selected.clone(),
                    needs_review: reading.needs_review,
                })
                .collect(),
            review_count: recognition.review_count,
            registration_failed: None,
            overlay,
        }
    }

    /// Reads a stored snapshot.
    pub fn load(path: &std::path::Path) -> Result<Self, Box<dyn std::error::Error>> {
        let text = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&text)?)
    }
}
