/* Copyright 2019 Kristofer Björnson
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/** @package TBTKcalc
 *  @file HartreeFock.h
 *  @brief Solves a Model using the Hartree-Fock method.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_HARTREE_FOCK_DIAGONALIZATION
#define COM_DAFER45_TBTK_HARTREE_FOCK_DIAGONALIZATION

#include "TBTK/Atom.h"
#include "TBTK/Solver/Diagonalizer.h"

namespace TBTK{
namespace Solver{

/** @brief Solves a Model using the Hartree-Fock method. */
class HartreeFock : public Diagonalizer{
public:
	class Callbacks :
		public HoppingAmplitude::AmplitudeCallback,
		public OverlapAmplitude::AmplitudeCallback
	{
	public:
		Callbacks();

		virtual std::complex<double> getHoppingAmplitude(
			const Index &to,
			const Index &from
		) const;

		virtual std::complex<double> getOverlapAmplitude(
			const Index &bra,
			const Index &ket
		) const;

		void setSolver(const HartreeFock &solver);
	private:
		const HartreeFock *solver;
	};

	class PositionedAtom : public Atom{
	public:
		PositionedAtom(const Atom &atom, const Vector3d &position);

		const Vector3d& getPosition() const;
	private:
		Vector3d position;
	};

	/** Constructs a Solver::HartreeFock. */
	HartreeFock();

	/** Destructor. */
	virtual ~HartreeFock();

	/** Get the total energy. */
	double getTotalEnergy();

	/** Get density matrix. */
	Matrix<std::complex<double>>& getDensityMatrix();

	/** Get density matrix. */
	const Matrix<std::complex<double>>& getDensityMatrix() const;

	/** Get nuclear centers. */
	std::vector<PositionedAtom>& getNuclearCenters();

	/** Get nuclear centers. */
	const std::vector<PositionedAtom>& getNuclearCenters() const;
private:
	Matrix<std::complex<double>> densityMatrix;

	std::vector<PositionedAtom> nuclearCenters;
};

inline Matrix<std::complex<double>>&
HartreeFock::getDensityMatrix(){
	return densityMatrix;
}

inline const Matrix<std::complex<double>>&
HartreeFock::getDensityMatrix() const{
	return densityMatrix;
}

inline void HartreeFock::Callbacks::setSolver(const HartreeFock &solver){
	this->solver = &solver;
}

inline HartreeFock::PositionedAtom::PositionedAtom(
	const Atom &atom,
	const Vector3d &position
) :
	Atom(atom),
	position(position)
{
}

inline const Vector3d& HartreeFock::PositionedAtom::getPosition() const{
	return position;
}

inline std::vector<HartreeFock::PositionedAtom>&
HartreeFock::getNuclearCenters(){
	return nuclearCenters;
}

inline const std::vector<HartreeFock::PositionedAtom>&
HartreeFock::getNuclearCenters() const{
	return nuclearCenters;
}

};	//End of namespace Solver
};	//End of namespace TBTK

#endif
