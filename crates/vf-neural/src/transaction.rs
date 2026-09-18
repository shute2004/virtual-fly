#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlasticityTransform {
    pub shift: f32,
    pub lo: f32,
    pub hi: f32,
}

impl PlasticityTransform {
    pub fn identity(weight_max: f32) -> Self {
        Self {
            shift: 0.0,
            lo: 0.0,
            hi: weight_max,
        }
    }

    pub fn observe_delta(&mut self, delta: f32, weight_max: f32) {
        self.shift += delta;
        self.lo = (self.lo + delta).clamp(0.0, weight_max);
        self.hi = (self.hi + delta).clamp(0.0, weight_max);
    }

    pub fn apply(self, weight: f32) -> f32 {
        (weight + self.shift).clamp(self.lo, self.hi)
    }

    pub fn is_identity(self, weight_max: f32) -> bool {
        self.shift == 0.0 && self.lo == 0.0 && self.hi == weight_max
    }
}

#[derive(Debug, Clone)]
pub struct PlasticityTransaction {
    pub source_weight_version: u64,
    pub transforms: Vec<PlasticityTransform>,
}

impl PlasticityTransaction {
    pub fn identity(edge_count: usize, source_weight_version: u64, weight_max: f32) -> Self {
        Self {
            source_weight_version,
            transforms: vec![PlasticityTransform::identity(weight_max); edge_count],
        }
    }

    pub fn apply_to(&self, weights: &mut [f32]) {
        assert_eq!(weights.len(), self.transforms.len());
        for (weight, transform) in weights.iter_mut().zip(&self.transforms) {
            *weight = transform.apply(*weight);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sequential(mut weight: f32, deltas: &[f32], weight_max: f32) -> f32 {
        for &delta in deltas {
            weight = (weight + delta).clamp(0.0, weight_max);
        }
        weight
    }

    #[test]
    fn identity_preserves_weight() {
        let transform = PlasticityTransform::identity(10.0);
        assert_eq!(transform.apply(3.5), 3.5);
    }

    #[test]
    fn transform_matches_additive_clamp_sequence() {
        let weight_max = 10.0;
        let cases: &[&[f32]] = &[
            &[],
            &[1.0],
            &[-1.0],
            &[20.0],
            &[-20.0],
            &[8.0, -3.0],
            &[-8.0, 3.0],
            &[20.0, -4.0, -20.0, 7.0],
            &[1.5, 2.5, -9.0, 4.0, 12.0, -1.0],
        ];
        let starts = [0.0, 0.25, 2.0, 5.0, 9.75, 10.0];

        for deltas in cases {
            let mut transform = PlasticityTransform::identity(weight_max);
            for &delta in *deltas {
                transform.observe_delta(delta, weight_max);
            }
            for start in starts {
                let expected = sequential(start, deltas, weight_max);
                let actual = transform.apply(start);
                assert!(
                    (expected - actual).abs() <= 1.0e-6,
                    "start={start} deltas={deltas:?} expected={expected} actual={actual} transform={transform:?}"
                );
            }
        }
    }

    #[test]
    fn stale_transaction_rebases_onto_latest_weight_without_overwrite() {
        let weight_max = 10.0;
        let mut stale = PlasticityTransform::identity(weight_max);
        stale.observe_delta(7.0, weight_max);
        stale.observe_delta(-2.0, weight_max);

        let latest_global = 1.0;
        assert_eq!(
            stale.apply(latest_global),
            sequential(latest_global, &[7.0, -2.0], weight_max)
        );
        assert_ne!(
            stale.apply(latest_global),
            sequential(9.0, &[7.0, -2.0], weight_max)
        );
    }
}
