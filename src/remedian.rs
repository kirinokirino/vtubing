use std::cmp::Ordering;

/// Computes the median of a slice of numbers.
/// If ordered is false, the slice will be sorted first.
fn median(lst: &mut [f32], ordered: bool) -> f32 {
    assert!(!lst.is_empty(), "median needs a non-empty list");

    if !ordered {
        lst.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    }

    let n = lst.len();
    let (p, q) = if n < 3 {
        (0, n - 1)
    } else {
        let mid = n / 2;
        if n % 2 == 0 {
            (mid, mid - 1)
        } else {
            (mid, mid)
        }
    };

    if p == q {
        lst[p]
    } else {
        (lst[p] + lst[q]) / 2.0
    }
}

/// Remedian structure for computing running medians of a stream of numbers
pub struct Remedian {
    all: Vec<f32>,
    k: usize,
    more: Option<Box<Remedian>>,
    cached_median: Option<f32>,
}

impl Remedian {
    /// Creates a new Remedian instance with optional initial values and bucket size
    pub fn new(k: usize) -> Self {
        Remedian {
            all: Vec::new(),
            k,
            more: None,
            cached_median: None,
        }
    }

    /// Adds a new value to the Remedian
    pub fn add(&mut self, x: f32) {
        self.cached_median = None;
        self.all.push(x);

        if self.all.len() == self.k {
            let med = self.median_prim();
            if self.more.is_none() {
                self.more = Some(Box::new(Remedian::new(self.k)));
            }
            if let Some(more) = &mut self.more {
                more.add(med);
            }
            self.all.clear();
        }
    }

    /// Returns the current median value
    pub fn median(&mut self) -> f32 {
        if let Some(more) = &mut self.more {
            more.median()
        } else {
            self.median_prim()
        }
    }

    /// Internal method to compute the median of the current bucket
    fn median_prim(&mut self) -> f32 {
        if self.cached_median.is_none() {
            let values = &mut self.all;
            self.cached_median = Some(median(values, false));
        }
        self.cached_median.unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_median() {
        let mut r = Remedian::new(64);
        for i in 0..1000 {
            r.add(i as f32);
        }
        assert!(r.median() > 450.0 && r.median() < 550.0);
    }
}
