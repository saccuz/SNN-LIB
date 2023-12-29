use rand::{Rng, SeedableRng};
use std::fmt::{Debug, Display, Formatter, Result};

pub trait Matrix {
    type T;
    fn zeroes(rows: usize, cols: usize) -> Self;
    fn random(rows: usize, cols: usize, diag: bool, seed: Option<u64>) -> Self;
    fn from(data: Vec<Vec<Self::T>>) -> Self;
    fn map(&mut self, function: &dyn Fn(Self::T) -> Self::T) -> Self;
    fn transpose(&self) -> Self;
}

#[derive(Clone)]
pub struct Input {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<u8>>,
}

impl Matrix for Input {
    type T = u8;
    fn zeroes(rows: usize, cols: usize) -> Input {
        Input {
            rows,
            cols,
            data: vec![vec![0; cols]; rows],
        }
    }

    fn random(rows: usize, cols: usize, diag: bool) -> Input {
        let mut rng = thread_rng();
        let mut res = Input::zeroes(rows, cols);

        for i in 0..rows {
            for j in 0..cols {
                if i == j && diag {
                    res.data[i][j] = 0;
                }
                res.data[i][j] = rng.gen_range(0..=1);
            }
        }
        return res;
    }

    fn from(data: Vec<Vec<u8>>) -> Input {
        let len = data[0].len();
        for t in &data {
            if t.len() != len {
                panic!(
                    "Invalid input, expected input to be shape= [{}], but got shape= [{}] instead",
                    len,
                    t.len()
                )
            }
        }
        Input {
            rows: data.len(),
            cols: data[0].len(),
            data,
        }
    }

    fn map(&mut self, function: &dyn Fn(u8) -> u8) -> Input {
        let res = <Input as Matrix>::from(
            (self.data)
                .clone()
                .into_iter()
                .map(|row| row.into_iter().map(|v| function(v)).collect())
                .collect(),
        );

        res
    }

    fn transpose(&self) -> Input {
        let mut res = Input::zeroes(self.cols, self.rows);

        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[j][i] = self.data[i][j];
            }
        }

        res
    }
}

impl Debug for Input {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(
            f,
            "Matrix {{\n{}\n}}",
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
impl Display for Input {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(
            f,
            "Matrix {{[\n{}] shape= ({}, {})}}",
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
