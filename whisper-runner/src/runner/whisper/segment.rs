use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DecodingResult {
    pub tokens: Vec<u32>,
    pub text: String,
    pub avg_logprob: f64,
    pub no_speech_prob: f64,
    pub temperature: f64,
    pub compression_ratio: f64,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Segment {
    pub start: f64,
    pub duration: f64,
    pub dr: DecodingResult,
}

/// whisper v3 timestamp tag inserted into text
static SEG_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"<\|([\d\.]+)\|>").unwrap());

// XXX vtt formating logic
impl Segment {
    /// consider next segment as same sentence if the gap is less than MERGE_NEXT_SECONDS
    const MERGE_NEXT_SECONDS: f64 = 0.6;
    fn vtt_header() -> String {
        "WEBVTT\n\n".to_owned()
    }
    pub fn build_vtt(segs: &Vec<Segment>) -> String {
        let mut vtt = Self::vtt_header();
        for seg in segs {
            vtt.push_str(&seg.vtt_cue());
        }
        vtt
    }
    pub fn has_timestamp_tokens(&self) -> bool {
        SEG_REGEX.is_match(self.text())
    }
    pub fn divide_by_ts_str(&self) -> Vec<Segment> {
        let divided =
            command_utils::text::SentenceSplitter::split_with_div_regex(&SEG_REGEX, self.text());
        if divided.len() == 1 {
            vec![self.clone()]
        } else {
            let mut segments = vec![];
            let mut start = self.start;
            let mut text: Option<String> = None;
            for txt in divided {
                if let Some((_, [tsstr])) = SEG_REGEX
                    .captures_iter(txt)
                    .map(|c| c.extract())
                    .collect::<Vec<(_, [&str; 1])>>()
                    .first()
                {
                    // ts string part
                    if let Ok(ts) = tsstr.parse::<f64>() {
                        if let Some(t) = text {
                            if ts + self.start < start {
                                // invalid ts warning
                                tracing::warn!(
                                    "Invalid ts string value: {} in segment start={}",
                                    txt,
                                    self.start
                                );
                            }
                            segments.push(Segment {
                                start,
                                duration: (ts + self.start - start),
                                dr: DecodingResult {
                                    tokens: vec![], // omit
                                    text: t,
                                    avg_logprob: self.dr.avg_logprob,
                                    no_speech_prob: self.dr.no_speech_prob,
                                    temperature: self.dr.temperature,
                                    compression_ratio: self.dr.compression_ratio,
                                },
                            });
                            text = None;
                        } else {
                            // begining tag
                            start = self.start + ts;
                        }
                    } else {
                        tracing::warn!("invalid ts string: {}, text: {}", tsstr, txt);
                    }
                } else {
                    // text part
                    text = Some(txt.to_string());
                }
            }
            segments // Return the segments vector
        }
    }
    pub fn divide_and_merge(&self, next: &Segment) -> Vec<Segment> {
        // divide self and next segment and  merge if necessary, self.divided.last() and next.divided.first()
        let mut segments = self.divide_by_ts_str();
        let mut next_segments = next.divide_by_ts_str();
        if let Some(last) = segments.last_mut() {
            if let Some(next_first) = next_segments.first_mut() {
                if (next_first.start - last.end()).abs() < Self::MERGE_NEXT_SECONDS {
                    // TODO 2-pass decoding for better transcription? (divided phrase wav is not good for decoding)
                    last.dr.text = format!("{} {}", last.dr.text, next_first.dr.text);
                    last.duration += next_first.duration;
                    next_segments.remove(0);
                }
            }
        }
        segments.append(&mut next_segments);
        segments
    }
    //extract floating number 3 chars as string from f64
    fn floating_num_str(num: f64) -> String {
        let mut num_str = format!("{:.03}", num);
        // trim before floating point
        num_str = num_str
            .trim_start_matches(['-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
            .to_string();
        num_str
    }
    pub fn text(&self) -> &str {
        &self.dr.text
    }
    pub fn start(&self) -> f64 {
        self.start
    }
    pub fn start_hms(&self) -> String {
        let secs = self.start as u32;
        let mins = secs / 60;
        let secs = secs % 60;
        let hours = mins / 60;
        let mins = mins % 60;
        format!(
            "{:02}:{:02}:{:02}{}",
            hours,
            mins,
            secs,
            Self::floating_num_str(self.start)
        )
    }
    pub fn end_hms(&self) -> String {
        let end = self.start + self.duration;
        let secs = (end) as u32;
        let mins = secs / 60;
        let secs = secs % 60;
        let hours = mins / 60;
        let mins = mins % 60;
        format!(
            "{:02}:{:02}:{:02}{}",
            hours,
            mins,
            secs,
            Self::floating_num_str(end)
        )
    }
    pub fn end(&self) -> f64 {
        self.start + self.duration
    }
    // https://www.w3.org/TR/webvtt1/
    pub fn vtt_cue(&self) -> String {
        let start = self.start_hms();
        let end = self.end_hms();
        let text = self.text();
        format!("{} --> {}\n{}\n\n", start, end, text)
    }
}

#[cfg(test)]
mod test {
    use crate::runner::whisper::segment::{DecodingResult, Segment};

    // test for Segment::floating_num
    #[test]
    fn test_segment_floating_num() {
        assert_eq!(Segment::floating_num_str(0.0001), ".000");
        assert_eq!(Segment::floating_num_str(1.0001), ".000");
        assert_eq!(Segment::floating_num_str(1f64), ".000");
        assert_eq!(Segment::floating_num_str(-1f64), ".000");
        assert_eq!(Segment::floating_num_str(0.0), ".000");
        assert_eq!(Segment::floating_num_str(20.10), ".100");
        assert_eq!(Segment::floating_num_str(-0.010), ".010");
        assert_eq!(Segment::floating_num_str(0.001), ".001");
        assert_eq!(Segment::floating_num_str(0.00001), ".000");
    }
    #[test]
    fn test_segment_divide_and_merge() {
        let text = r#"<|7.54|> All the time.<|12.34|><|12.98|> Interviews.<|15.50|><|16.04|> I'm your host.<|17.74|><|19.46|> The idea<|24.38|><|24.38|>"#;
        let segment = Segment {
            start: 120.0,
            duration: 30.0,
            dr: DecodingResult {
                tokens: vec![],
                text: text.to_string(),
                avg_logprob: 0.0,
                no_speech_prob: 0.0,
                temperature: 0.0,
                compression_ratio: 0.0,
            },
        };
        let next_text = r#"<|0.00|> Next time.<|12.34|><|28.34|>"#;
        let next_segment = Segment {
            start: 150.0,
            duration: 30.0,
            dr: DecodingResult {
                tokens: vec![],
                text: next_text.to_string(),
                avg_logprob: 0.0,
                no_speech_prob: 0.0,
                temperature: 0.0,
                compression_ratio: 0.0,
            },
        };
        let expected = vec![
            Segment {
                start: 127.54,
                duration: 4.80,
                dr: DecodingResult {
                    tokens: vec![],
                    text: " All the time.".to_string(),
                    avg_logprob: 0.0,
                    no_speech_prob: 0.0,
                    temperature: 0.0,
                    compression_ratio: 0.0,
                },
            },
            Segment {
                start: 132.98,
                duration: 2.52,
                dr: DecodingResult {
                    tokens: vec![],
                    text: " Interviews.".to_string(),
                    avg_logprob: 0.0,
                    no_speech_prob: 0.0,
                    temperature: 0.0,
                    compression_ratio: 0.0,
                },
            },
            Segment {
                start: 136.04,
                duration: 1.70,
                dr: DecodingResult {
                    tokens: vec![],
                    text: " I'm your host.".to_string(),
                    avg_logprob: 0.0,
                    no_speech_prob: 0.0,
                    temperature: 0.0,
                    compression_ratio: 0.0,
                },
            },
            Segment {
                start: 139.46,
                duration: 4.92,
                dr: DecodingResult {
                    tokens: vec![],
                    text: " The idea".to_string(),
                    avg_logprob: 0.0,
                    no_speech_prob: 0.0,
                    temperature: 0.0,
                    compression_ratio: 0.0,
                },
            },
            Segment {
                start: 150.0,
                duration: 12.34,
                dr: DecodingResult {
                    tokens: vec![],
                    text: " Next time.".to_string(),
                    avg_logprob: 0.0,
                    no_speech_prob: 0.0,
                    temperature: 0.0,
                    compression_ratio: 0.0,
                },
            },
        ];
        assert!(segment
            .divide_and_merge(&next_segment)
            .iter()
            .zip(expected.iter())
            .all(|(a, b)| {
                let test = (a.start - b.start).abs() < 0.00000001 // avoid rounding error
                    && (a.duration - b.duration).abs() < 0.00000001
                    && a.text() == b.text();
                if !test {
                    panic!("== segment: {:?}, expected: {:?}", a, b);
                }
                test
            }));

        // single ts only text segment
        let ts_text = r#""#;
        let segment = Segment {
            start: 120.0,
            duration: 30.0,
            dr: DecodingResult {
                tokens: vec![],
                text: ts_text.to_string(),
                avg_logprob: 0.0,
                no_speech_prob: 0.0,
                temperature: 0.0,
                compression_ratio: 0.0,
            },
        };
        let expected = [Segment {
            start: 150.0,
            duration: 12.34,
            dr: DecodingResult {
                tokens: vec![],
                text: " Next time.".to_string(),
                avg_logprob: 0.0,
                no_speech_prob: 0.0,
                temperature: 0.0,
                compression_ratio: 0.0,
            },
        }];
        assert!(segment
            .divide_and_merge(&next_segment)
            .iter()
            .zip(expected.iter())
            .all(|(a, b)| {
                let test = (a.start - b.start).abs() < 0.00000001
                    && (a.duration - b.duration).abs() < 0.00000001
                    && a.text() == b.text();
                if !test {
                    panic!("== segment: {:?}, expected: {:?}", a, b);
                }
                test
            }));
    }
    #[test]
    fn test_segment_divide_and_merge_merge() {
        // merge text segment
        let txt = "hogefugaaa.";
        let ts_text = format!("<|20.00|>{txt}<|29.80|>");
        let segment = Segment {
            start: 120.0,
            duration: 30.0,
            dr: DecodingResult {
                tokens: vec![],
                text: ts_text,
                avg_logprob: 0.0,
                no_speech_prob: 0.0,
                temperature: 0.0,
                compression_ratio: 0.0,
            },
        };
        let next_text = r#"<|0.00|> Next time.<|12.34|><|28.34|>"#;
        let next_segment = Segment {
            start: 150.0,
            duration: 30.0,
            dr: DecodingResult {
                tokens: vec![],
                text: next_text.to_string(),
                avg_logprob: 0.0,
                no_speech_prob: 0.0,
                temperature: 0.0,
                compression_ratio: 0.0,
            },
        };

        let expected = [Segment {
            start: 140.0,
            duration: 9.80 + 12.34,
            dr: DecodingResult {
                tokens: vec![],
                text: format!("{txt}  Next time."),
                avg_logprob: 0.0,
                no_speech_prob: 0.0,
                temperature: 0.0,
                compression_ratio: 0.0,
            },
        }];
        let res = segment.divide_and_merge(&next_segment);
        assert!(res.iter().zip(expected.iter()).all(|(a, b)| {
            let test = (a.start - b.start).abs() < 0.00000001
                && (a.duration - b.duration).abs() < 0.00000001
                && a.text() == b.text();
            if !test {
                panic!("== segment: {:?}, expected: {:?}", a, b);
            }
            test
        }));
    }
}
