use bellman::groth16;
use bellman::{Circuit, ConstraintSystem, SynthesisError};
use bls12_381_plus::{Bls12, Scalar};
use rand::rngs::OsRng;

mod consts {
    use bls12_381_plus::Scalar;

    pub const ONE: Scalar = Scalar::ONE;
    pub const TWO: Scalar = ONE.add(&ONE);
    pub const THREE: Scalar = TWO.add(&ONE);
    pub const FOUR: Scalar = THREE.add(&ONE);
}

fn main() {
    let params = groth16::generate_random_parameters::<Bls12, _, _>(
        EqualCircuit {
            secret: consts::ONE,
            public: consts::ONE,
        },
        &mut OsRng,
    )
    .unwrap();

    let vk = groth16::prepare_verifying_key(&params.vk);

    let proof = groth16::create_random_proof(
        EqualCircuit {
            secret: consts::THREE,
            public: consts::THREE,
        },
        &params,
        &mut OsRng,
    )
    .unwrap();

    assert!(groth16::verify_proof(&vk, &proof, &[consts::THREE]).is_ok());
    assert!(groth16::verify_proof(&vk, &proof, &[consts::FOUR]).is_err());
}

#[derive(Debug, Copy, Clone)]
struct EqualCircuit {
    secret: Scalar,
    public: Scalar,
}

impl Circuit<Scalar> for EqualCircuit {
    fn synthesize<CS: ConstraintSystem<Scalar>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
        let one = CS::one();

        let public = cs.alloc_input(|| "public", || Ok(self.public))?;
        let secret = cs.alloc(|| "secret", || Ok(self.secret))?;

        cs.enforce(
            || "compare public and private values",
            |lc| lc + one,
            |lc| lc + public,
            |lc| lc + secret,
        );

        Ok(())
    }
}
