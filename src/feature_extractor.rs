use crate::feature::Feature;

pub struct FeatureExtractor {
    eye_l: Feature,
    eye_r: Feature,
    eyebrow_updown_l: Feature,
    eyebrow_updown_r: Feature,
    eyebrow_quirk_l: Feature,
    eyebrow_quirk_r: Feature,
    eyebrow_steepness_l: Feature,
    eyebrow_steepness_r: Feature,
    mouth_corner_updown_l: Feature,
    mouth_corner_updown_r: Feature,
    mouth_corner_inout_l: Feature,
    mouth_corner_inout_r: Feature,
    mouth_open: Feature,
    mouth_wide: Feature,
}

impl FeatureExtractor {
    pub fn new(max_feature_updates: i32) -> Self {
        FeatureExtractor {
            eye_l: Feature::new(0.15, 0.2, 0.15, 0.001, max_feature_updates),
            eye_r: Feature::new(0.15, 0.2, 0.15, 0.001, max_feature_updates),
            eyebrow_updown_l: Feature::new(0.15, 0.2, 0.15, 0.001, max_feature_updates),
            eyebrow_updown_r: Feature::new(0.15, 0.2, 0.15, 0.001, max_feature_updates),
            eyebrow_quirk_l: Feature::new(0.05, 0.2, 0.15, 0.001, max_feature_updates),
            eyebrow_quirk_r: Feature::new(0.05, 0.2, 0.15, 0.001, max_feature_updates),
            eyebrow_steepness_l: Feature::new(0.05, 0.2, 0.15, 0.001, max_feature_updates),
            eyebrow_steepness_r: Feature::new(0.05, 0.2, 0.15, 0.001, max_feature_updates),
            mouth_corner_updown_l: Feature::new(0.15, 0.2, 0.15, 0.001, max_feature_updates),
            mouth_corner_updown_r: Feature::new(0.15, 0.2, 0.15, 0.001, max_feature_updates),
            mouth_corner_inout_l: Feature::new(0.02, 0.2, 0.15, 0.001, max_feature_updates),
            mouth_corner_inout_r: Feature::new(0.02, 0.2, 0.15, 0.001, max_feature_updates),
            mouth_open: Feature::new(0.15, 0.2, 0.15, 0.001, max_feature_updates),
            mouth_wide: Feature::new(0.15, 0.2, 0.15, 0.001, max_feature_updates),
        }
    }
}
