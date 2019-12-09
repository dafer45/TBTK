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

/// @cond TBTK_FULL_DOCUMENTATION
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
	/** A callback class that acts as mediator between the Model and the
	 *  Solver::HartreeFock. An object of this class should be passed to
	 *  the Model to generate the HoppingAmplitudeSet and
	 *  OverlapAmplitudeSet from a set of basis functions. These callbacks
	 *  allows the Model to stay up to date with the solvers
	 *  self-consistenly calculated parameters. */
	class Callbacks :
		public HoppingAmplitude::AmplitudeCallback,
		public OverlapAmplitude::AmplitudeCallback
	{
	public:
		/** Constructor. */
		Callbacks();

		/** Implements
		 *  HoppingAmplitude::AmplitudeCallback::getHoppingAmplitude().
		 */
		virtual std::complex<double> getHoppingAmplitude(
			const Index &to,
			const Index &from
		) const;

		/** Implements
		 *  OverlapAmplitude::AmplitudeCallback::getOverlapAmplitude().
		 */
		virtual std::complex<double> getOverlapAmplitude(
			const Index &bra,
			const Index &ket
		) const;

		/** Set the Hartree-Fock solver that the callbacks uses to
		 *  update the Model parameters.
		 *
		 *  @param solver The Hartree-Fock solver that the callbacks
		 *  use to update the Model parameters. */
		void setSolver(const HartreeFock &solver);
	private:
		/** The Hartree-Fock solver that the callbacks use to update
		 *  the Model parameters. */
		const HartreeFock *solver;
	};

	/** Constructs a Solver::HartreeFock. */
	HartreeFock();

	/** Destructor. */
	virtual ~HartreeFock();

	/** Set the occupation number.
	 *
	 *  @param occupationNumber The occupation number. */
	void setOccupationNumber(unsigned int occupationNumber);

	/** Get the total energy.
	 *
	 *  @return The total energy. */
	double getTotalEnergy() const;

	/** Add a nuclear center.
	 *
	 *  @param atom The atom type of the nucleus.
	 *  @param position The nucleus position. */
	void addNuclearCenter(const Atom &atom, const Vector3d &position);

	/** Run the calculation. */
	void run();
private:
	/** Helper class for storing a nuclear center. */
	class PositionedAtom : public Atom{
	public:
		/** Constructor.
		 *
		 *  @param atom The atom type of the nucleus.
		 *  @param psoition The nucleus position. */
		PositionedAtom(const Atom &atom, const Vector3d &position);

		/** Get position.
		 *
		 *  @return The nucleus position. */
		const Vector3d& getPosition() const;
	private:
		//Position
		Vector3d position;
	};

	/** The basis states. */
	std::vector<const AbstractState*> basisStates;

	/** The density matrix. */
	Matrix<std::complex<double>> densityMatrix;

	/** The nuclear centers. */
	std::vector<PositionedAtom> nuclearCenters;

	/** The occupation number. */
	unsigned int occupationNumber;

	/** The total energy. */
	double totalEnergy;

	/** The slef-consistency callback. */
	class SelfConsistencyCallback :
		public Diagonalizer::SelfConsistencyCallback
	{
	public:
		/** Constructor.
		 *
		 *  @param solver The solver that is associated with the
		 *  callback. */
		SelfConsistencyCallback(HartreeFock &solver);

		/** Implements
		 *  Solver::Diagonalizer::SelfConsistencyCallaback::selfConsistencyCallback().
		 */
		virtual bool selfConsistencyCallback(
			Diagonalizer &diagonalizer
		);
	private:
		/** The solver that is associated with the callback. */
		HartreeFock &solver;
	} selfConsistencyCallback;

	/** Get the density matrix. */
	const Matrix<std::complex<double>>& getDensityMatrix() const;

	/** Get the nuclear centers. */
	const std::vector<PositionedAtom>& getNuclearCenters() const;

	/** Calculate the total energy. */
	void calculateTotalEnergy();
};

inline void HartreeFock::setOccupationNumber(unsigned int occupationNumber){
	this->occupationNumber = occupationNumber;
}

inline double HartreeFock::getTotalEnergy() const{
	return totalEnergy;
}

inline const Matrix<std::complex<double>>&
HartreeFock::getDensityMatrix() const{
	return densityMatrix;
}

inline void HartreeFock::addNuclearCenter(
	const Atom &atom,
	const Vector3d &position
){
	nuclearCenters.push_back(PositionedAtom(atom, position));
}

inline const std::vector<HartreeFock::PositionedAtom>&
HartreeFock::getNuclearCenters() const{
	return nuclearCenters;
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

};	//End of namespace Solver
};	//End of namespace TBTK

#endif
/// @endcond
