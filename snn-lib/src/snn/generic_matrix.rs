use rand::{Rng, SeedableRng};
use std::fmt::{Debug, Display, Formatter, Result};
use rand::distributions::uniform::SampleUniform;
use rand::prelude::StdRng;

#[derive(Clone)]
pub struct MatrixG<T : Default + Clone + Copy + SampleUniform + PartialOrd + Display> {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<T>>,
}

impl<T : Default + Clone + Copy + SampleUniform + PartialOrd + Display> MatrixG<T> {
    fn zeroes(rows: usize, cols: usize) -> MatrixG<T> {
        MatrixG {
            rows,
            cols,
            data: vec![vec![T::default(); cols]; rows],
        }
    }

    pub fn random(rows: usize, cols: usize, diag: bool, seed: Option<u64>, a: T, b: T) -> MatrixG<T> {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy()
        };

        let mut res = MatrixG::zeroes(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                if diag && i == j {
                    res.data[i][j] = T::default();
                }
                else { res.data[i][j] = rng.gen_range(a..=b); }
            }
        }
        res
    }

    pub fn from(data: Vec<Vec<T>>) -> MatrixG<T> {
        let len = data[0].len();
        for t in &data {
            if t.len() != len {
                panic!(
                    "Invalid MatrixG, expected MatrixG to be shape= [{}], but got shape= [{}] instead",
                    len,
                    t.len()
                )
            }
        }
        MatrixG {
            rows: data.len(),
            cols: data[0].len(),
            data,
        }
    }

    // TODO: if needed implement this
    /*fn transpose(&self) -> MatrixG {
        let mut res = MatrixG::zeroes(self.cols, self.rows);

        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[j][i] = self.data[i][j];
            }
        }

        res
    }*/
}

impl<T: Default + Clone + Copy + SampleUniform + PartialOrd + Display> Debug for MatrixG<T> {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(
            f,
            "MatrixG<{}> {{\n{}\n}}",
            std::any::type_name::<T>(),
            (&self.data)
                .into_iter()
                .map(|row| "  ".to_string()
                    + &row
                    .into_iter()
                    .map(|value| value.to_string())
                    .collect::<Vec<String>>()
                    .join(" "))
                .collect::<Vec<String>>()
                .join("\n")
        )
    }
}
impl<T: Default + Clone + Copy + SampleUniform + PartialOrd + Display> Display for MatrixG<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(
            f,
            "MatrixG<{}> {{[\n{}] shape= ({}, {})}}",
            std::any::type_name::<T>(),
            (&self.data)
                .into_iter()
                .map(|row| "  ".to_string()
                    + &row
                    .into_iter()
                    .map(|value| value.to_string())
                    .collect::<Vec<String>>()
                    .join(" "))
                .collect::<Vec<String>>()
                .join("\n"),
            self.rows,
            self.cols
        )
    }
}
