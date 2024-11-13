use crate::remedian::Remedian;

pub struct Feature {
    median: Remedian,
    min: Option<f32>,
    max: Option<f32>,
    hard_min: Option<f32>,
    hard_max: Option<f32>,
    threshold: f32,
    alpha: f32,
    hard_factor: f32,
    decay: f32,
    last: f32,
    current_median: f32,
    update_count: i32,
    max_feature_updates: i32,
    first_seen: f32,
    updating: bool,
}

impl Feature {
    pub fn new(
        threshold: f32,
        alpha: f32,
        hard_factor: f32,
        decay: f32,
        max_feature_updates: i32,
    ) -> Self {
        Feature {
            median: Remedian::new(64),
            min: None,
            max: None,
            hard_min: None,
            hard_max: None,
            threshold,
            alpha,
            hard_factor,
            decay,
            last: 0.0,
            current_median: 0.0,
            update_count: 0,
            max_feature_updates,
            first_seen: -1.0,
            updating: true,
        }
    }

    pub fn update(&mut self, x: f32, now: f32) -> f32 {
        if self.max_feature_updates > 0 && self.first_seen == -1.0 {
            self.first_seen = now;
        }
        
        let new = self.update_state(x, now);
        let filtered = self.last * self.alpha + new * (1.0 - self.alpha);
        self.last = filtered;
        filtered
    }

    fn update_state(&mut self, x: f32, now: f32) -> f32 {
        let updating = self.updating && (self.max_feature_updates == 0 
            || now - self.first_seen < self.max_feature_updates as f32);

        if updating {
            self.median.add(x);
            self.current_median = self.median.median();
        } else {
            self.updating = false;
        }
        let median = self.current_median;

        // Handle min case
        if self.min.is_none() {
            if x < median && (median - x) / median > self.threshold {
                if updating {
                    self.min = Some(x);
                    self.hard_min = Some(x + self.hard_factor * (median - x));
                }
                return -1.0;
            }
            return 0.0;
        } else if x < self.min.unwrap() {
            if updating {
                self.min = Some(x);
                self.hard_min = Some(x + self.hard_factor * (median - x));
            }
            return -1.0;
        }

        // Handle max case
        if self.max.is_none() {
            if x > median && (x - median) / median > self.threshold {
                if updating {
                    self.max = Some(x);
                    self.hard_max = Some(x - self.hard_factor * (x - median));
                }
                return 1.0;
            }
            return 0.0;
        } else if x > self.max.unwrap() {
            if updating {
                self.max = Some(x);
                self.hard_max = Some(x - self.hard_factor * (x - median));
            }
            return 1.0;
        }

        // Update min/max if updating
        if updating {
            if let Some(min) = self.min {
                if let Some(hard_min) = self.hard_min {
                    if min < hard_min {
                        self.min = Some(hard_min * self.decay + min * (1.0 - self.decay));
                    }
                }
            }

            if let Some(max) = self.max {
                if let Some(hard_max) = self.hard_max {
                    if max > hard_max {
                        self.max = Some(hard_max * self.decay + max * (1.0 - self.decay));
                    }
                }
            }
        }

        // Calculate final value
        if x < median {
            if let Some(min) = self.min {
                -(1.0 - (x - min) / (median - min))
            } else {
                0.0
            }
        } else if x > median {
            if let Some(max) = self.max {
                (x - median) / (max - median)
            } else {
                0.0
            }
        } else {
            0.0
        }
    }
}
