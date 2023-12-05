//Create a matrix struct to manage a Vec<Vec<T>> matrix but do it in a single struct without using a trait.

use std::ops::{Add, Mul, Sub};
use rand::{Rng, thread_rng};
use rand::distributions::{Distribution, Standard};

#[derive(Clone)]
pub struct D {
    dim: usize,
}
impl D{
    pub fn new(dim: usize) -> Self {
        D {
            dim
        }
    }
}

pub type A = D;
trait Dim {
    fn dim(&self) -> usize;
}

impl Dim for D{
    fn dim(&self) -> usize {
        self.dim
    }
}

//Make a struct VMatrix<T,R,C> that contains a Vec<Vec<T>> and the number of rows and columns using Dim trait
pub struct VMatrix<T, R: Dim, C: Dim> {
    rows: R,
    cols: C,
    data: Vec<Vec<T>>,
}

//Make the new function for VMatrix

impl<T, R: Dim, C: Dim> VMatrix<T, R, C> {
    pub fn new() -> Self {
        VMatrix {
            rows: R::dim(),
            cols: C::dim(),
            data: vec![vec![T::default(); C::dim()]; R::dim()],
        }
    }
}

//implement Distribution for VMatrix<T,R,C> where T is u8
impl<R: Dim, C: Dim> Distribution<VMatrix<u8, R, C>> for Standard {
    fn sample<X: Rng + ?Sized>(&self, rng: &mut X) -> VMatrix<u8, R, C> {
        let mut res = VMatrix::<u8, R, C>::new();
        for i in 0..R::dim() {
            for j in 0..C::dim() {
                res.data[i][j] = rng.gen_range(0..=1);
            }
        }
        res
    }
}

//implement Distribution for VMatrix<T,R,C> where T is f64
impl<R: Dim, C: Dim> Distribution<VMatrix<f64, R, C>> for Standard {
    fn sample<X: Rng + ?Sized>(&self, rng: &mut X) -> VMatrix<f64, R, C> {
        let mut res = VMatrix::<f64, R, C>::new();
        for i in 0..R::dim() {
            for j in 0..C::dim() {
                res.data[i][j] = rng.gen_range(0.01..1.0);
            }
        }
        res
    }
}

//implement display for VMatrix




pub struct CMatrix<T, const ROW: usize, const COL: usize>{
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<T>>,
}


impl<T, const ROW: usize, const COL: usize> CMatrix<T, ROW, COL>
where
    T: Copy + Default,
{
    pub fn new() -> Self {
        CMatrix {
            rows: ROW,
            cols: COL,
            data: vec![vec![T::default(); COL]; ROW],
        }
    }
}
//Distribution implementation for u8 (Spikes) and f64 (Weights)
impl<const ROW: usize, const COL: usize> Distribution<CMatrix<f64, ROW, COL>> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> CMatrix<f64, ROW, COL> {
        let mut res = CMatrix::<f64, ROW, COL>::new();
        for i in 0..ROW {
            for j in 0..COL {
                res.data[i][j] = rng.gen_range(0.01..1.0);
            }
        }
        res
    }
}

impl<const ROW: usize, const COL: usize> Distribution<CMatrix<u8, ROW, COL>> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> CMatrix<u8, ROW, COL> {
        let mut res = CMatrix::<u8, ROW, COL>::new();
        for i in 0..ROW {
            for j in 0..COL {
                res.data[i][j] = rng.gen_range(0..=1);
            }
        }
        res
    }
}

//Random function for CMatrix
impl<T, const ROW: usize, const COL: usize> CMatrix<T, ROW, COL>
where
    T: Copy + Default + Add<Output = T> + Mul<Output = T> + Sub<Output = T> , Standard: Distribution<CMatrix<T, ROW, COL>>
{
    pub fn random() -> Self {
        let mut rng = thread_rng();
        let res = rng.gen::<CMatrix<T, ROW, COL>>();
        res
    }
}

//implement Display for CMatrix<T, ROW, COL>
impl<T, const ROW: usize, const COL: usize> std::fmt::Display for CMatrix<T, ROW, COL>
where
    T: Copy + Default + Add<Output = T> + Mul<Output = T> + Sub<Output = T> + std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut res = String::new();
        for i in 0..ROW {
            for j in 0..COL {
                res.push_str(&format!("{} ", self.data[i][j]));
            }
            res.push_str("\n");
        }
        write!(f, "{}", res)
    }
}

//Capire se eliminare tutto il codice sotto
pub struct Matrix<T> {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<T>>,
}

impl<T> Matrix<T>
where
    T: Copy + Default,
{
    pub fn new(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![vec![T::default(); cols]; rows],
        }
    }

    pub fn from(data: Vec<Vec<T>>) -> Self {
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
        Matrix {
            rows: data.len(),
            cols: data[0].len(),
            data,
        }
    }

    pub fn transpose(&self) -> Self {
        let mut res = Matrix::new(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[j][i] = self.data[i][j];
            }
        }
        res
    }

    pub fn map(&mut self, function: &dyn Fn(T) -> T) -> Self {
        let mut res = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[i][j] = function(self.data[i][j]);
            }
        }
        res
    }
}

impl<T> Matrix<T>
where
    T: Copy + Default + Add<Output = T> + Mul<Output = T> + Sub<Output = T>,
{
    pub fn dot(&self, other: &Matrix<T>) -> Self {
        if self.cols != other.rows {
            panic!(
                "Invalid input, expected input to be shape= [{}], but got shape= [{}] instead",
                self.cols, other.rows
            )
        }
        let mut res = Matrix::new(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                for k in 0..self.cols {
                    res.data[i][j] = res.data[i][j] + self.data[i][k] * other.data[k][j];
                }
            }
        }
        res
    }
}

impl<T> Matrix<T>
where
    T: Copy + Default + Add<Output = T> + Mul<Output = T> + Sub<Output = T>,
{
    pub fn hadamard(&self, other: &Matrix<T>) -> Self {
        if self.rows != other.rows || self.cols != other.cols {
            panic!(
                "Invalid input, expected input to be shape= [{}], but got shape= [{}] instead",
                self.cols, other.rows
            )
        }
        let mut res = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[i][j] = self.data[i][j] * other.data[i][j];
            }
        }
        res
    }
}

impl<T> Matrix<T>
where
    T: Copy + Default + Add<Output = T> + Mul<Output = T> + Sub<Output = T>,
{
    pub fn add(&self, other: &Matrix<T>) -> Self {
        if self.rows != other.rows || self.cols != other.cols {
            panic!(
                "Invalid input, expected input to be shape= [{}], but got shape= [{}] instead",
                self.cols, other.rows
            )
        }
        let mut res = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[i][j] = self.data[i][j] + other.data[i][j];
            }
        }
        res
    }
}

impl<T> Matrix<T>
where
    T: Copy + Default + Add<Output = T> + Mul<Output = T> + Sub<Output = T>,
{
    pub fn sub(&self, other: &Matrix<T>) -> Self {
        if self.rows != other.rows || self.cols != other.cols {
            panic!(
                "Invalid input, expected input to be shape= [{}], but got shape= [{}] instead",
                self.cols, other.rows
            )
        }
        let mut res = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[i][j] = self.data[i][j] - other.data[i][j];
            }
        }
        res
    }
}

impl<T> Matrix<T>
where
    T: Copy + Default + Add<Output = T> + Mul<Output = T> + Sub<Output = T>,
{
    pub fn scalar_mul(&self, scalar: T) -> Self {
        let mut res = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[i][j] = self.data[i][j] * scalar;
            }
        }
        res
    }
}

impl<T> Matrix<T>
where
    T: Copy + Default + Add<Output = T> + Mul<Output = T> + Sub<Output = T>,
{
    pub fn scalar_add(&self, scalar: T) -> Self {
        let mut res = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[i][j] = self.data[i][j] + scalar;
            }
        }
        res
    }
}

impl<T> Matrix<T>
where
    T: Copy + Default + Add<Output = T> + Mul<Output = T> + Sub<Output = T>,
{
    pub fn scalar_sub(&self, scalar: T) -> Self {
        let mut res = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                res.data[i][j] = self.data[i][j] - scalar;
            }
        }
        res
    }
}

impl Matrix<f64> {
    pub fn random(rows: usize, cols: usize, diag: bool) -> Self {
        let mut rng = thread_rng();
        let mut res = Matrix::new(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                let rnd_number = rng.gen_range(0.01..1.0);
                if diag {
                    if i == j {
                        res.data[i][j] = 0.0;
                    } else {
                        res.data[i][j] = -rnd_number;
                    }
                } else {
                    res.data[i][j] = rnd_number;
                }
            }
        }
        res
    }
}

impl Matrix<u8> {
    pub fn random(rows: usize, cols: usize, diag: bool) -> Self {
        let mut rng = thread_rng();
        let mut res = Matrix::new(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                let rnd_number = rng.gen_range(0..=1);
                if diag {
                    if i == j {
                        res.data[i][j] = 0;
                    } else {
                        res.data[i][j] = rnd_number;
                    }
                } else {
                    res.data[i][j] = rnd_number;
                }
            }
        }
        res
    }
}

//implement Display for Matrix

impl<T> std::fmt::Display for Matrix<T>
where
    T: Copy + Default + Add<Output = T> + Mul<Output = T> + Sub<Output = T> + std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut res = String::new();
        for i in 0..self.rows {
            for j in 0..self.cols {
                res.push_str(&format!("{} ", self.data[i][j]));
            }
            res.push_str("\n");
        }
        write!(f, "{}", res)
    }
}



