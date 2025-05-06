use colorgrad::{self, Gradient};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::FromPyObject;

pub enum Colormap {
    Inferno,
    Viridis,
    Plasma,
    Magma,
    Greys,
    Blues,
    Greens,
    Reds,
    Purples,
    Oranges,
}

impl<'py> FromPyObject<'py> for Colormap {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let name: String = ob.extract()?;
        Colormap::from_name(&name)
            .ok_or_else(|| PyValueError::new_err(format!("Invalid colormap name: {}", name)))
    }
}

impl<'py> IntoPyObject<'py> for Colormap {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = std::convert::Infallible;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let s = match self {
            Colormap::Inferno => "inferno",
            Colormap::Viridis => "viridis",
            Colormap::Plasma => "plasma",
            Colormap::Magma => "magma",
            Colormap::Greys => "greys",
            Colormap::Blues => "blues",
            Colormap::Greens => "greens",
            Colormap::Reds => "reds",
            Colormap::Purples => "purples",
            Colormap::Oranges => "oranges",
        };
        Ok(s.into_pyobject(py)?.into_any())
    }
}

impl Colormap {
    pub fn to_gradient(&self) -> impl Gradient {
        match self {
            Colormap::Inferno => colorgrad::preset::inferno(),
            Colormap::Viridis => colorgrad::preset::viridis(),
            Colormap::Plasma => colorgrad::preset::plasma(),
            Colormap::Magma => colorgrad::preset::magma(),
            Colormap::Greys => colorgrad::preset::greys(),
            Colormap::Blues => colorgrad::preset::blues(),
            Colormap::Greens => colorgrad::preset::greens(),
            Colormap::Reds => colorgrad::preset::reds(),
            Colormap::Purples => colorgrad::preset::purples(),
            Colormap::Oranges => colorgrad::preset::oranges(),
        }
    }
    pub fn from_name(name: &str) -> Option<Colormap> {
        match name {
            "inferno" => Some(Colormap::Inferno),
            "viridis" => Some(Colormap::Viridis),
            "plasma" => Some(Colormap::Plasma),
            "magma" => Some(Colormap::Magma),
            "greys" => Some(Colormap::Greys),
            "blues" => Some(Colormap::Blues),
            "greens" => Some(Colormap::Greens),
            "reds" => Some(Colormap::Reds),
            "purples" => Some(Colormap::Purples),
            "oranges" => Some(Colormap::Oranges),
            _ => None,
        }
    }
}

/// Precomputes 256 RGB colours for the given colormap.
pub fn precompute_colormap(cmap: &Colormap) -> Vec<[u8; 3]> {
    let grad = cmap.to_gradient();
    (0..256)
        .map(|i| {
            let rgba = grad.at(i as f32 / 255.0).to_rgba8();
            [rgba[0], rgba[1], rgba[2]]
        })
        .collect()
}
