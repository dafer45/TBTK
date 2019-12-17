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

/** @file Sto3g.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Sto3g.h"
#include "TBTK/TBTKMacros.h"

using namespace std;

namespace TBTK{

Sto3g::Sto3g(
	double slaterExponent,
	const vector<double> &coordinates,
	const Index &index,
	int spinIndex
) :
	AbstractState(StateID::STO3G)
{
	setCoordinates(coordinates);
	setIndex(index);
	setExtent(numeric_limits<double>::infinity());

	gaussianExponents[0] = 0.109818*pow(slaterExponent, 2);
	gaussianExponents[1] = 0.405771*pow(slaterExponent, 2);
	gaussianExponents[2] = 2.22766*pow(slaterExponent, 2);

	this->spinIndex = spinIndex;
}

complex<double> Sto3g::getOverlap(const AbstractState &ket) const{
	//Only overlaps between Sto3g states are supported.
	TBTKAssert(
		ket.getStateID() == StateID::STO3G,
		"Sto3g::getOverlap()",
		"Ket state with StateID '" << ket.getStateID() << "' is not"
		<< " supported.",
		""
	);
	const Sto3g &ketSto3g = (const Sto3g&)ket;

	//Confirm that the states either both have or both do not have spin
	//indices. In the former case, return zero if the spin indices are
	//different.
	int braSpinIndex = spinIndex;
	int ketSpinIndex = ketSto3g.spinIndex;
	TBTKAssert(
		(braSpinIndex == -1 && ketSpinIndex == -1)
		|| (braSpinIndex != -1 && ketSpinIndex != -1),
		"Sto3g::getOverlap()",
		"Bra and ket state must either both have or both not have a"
		<< "spin index.",
		""
	);
	if(braSpinIndex != -1){
		if(getIndex()[braSpinIndex] != ketSto3g.getIndex()[
				ketSpinIndex
			]
		){
			return 0;
		}
	}

	//The implemented equations assumes that distances are given in terms
	//of the Bohr radius. Therefore distances have to be converted to this
	//unit.
	double bohrRadius = UnitHandler::getConstantInNaturalUnits("a_0");

	//Calculate the distance between the two states.
	Vector3d braR = Vector3d(getCoordinates())/bohrRadius;
	Vector3d ketR = Vector3d(ketSto3g.getCoordinates())/bohrRadius;
	double deltaR = (braR - ketR).norm();

	//The main algorithm. See Appendix A in the book Modern Quantum
	//Chemistry: Introduction to Advanced Electronic Structure Theory by
	//A Szabo and NS Ostlund.
	double overlap = 0;
	for(unsigned int n = 0; n < 3; n++){
		double alpha = gaussianExponents[n];
		double normA = pow(2*alpha/M_PI, 3/4.);
		double dA = contractionCoefficients[n];
		for(unsigned int np = 0; np < 3; np++){
			double beta = ketSto3g.gaussianExponents[np];
			double normB = pow(2*beta/M_PI, 3/4.);
			double dB = ketSto3g.contractionCoefficients[np];

			overlap += dA*dB*normA*normB*pow(
				M_PI/(alpha + beta),
				3/2.
			)*exp(-(alpha*beta/(alpha + beta))*pow(deltaR, 2));
		}
	}

	return overlap;
}

complex<double> Sto3g::getMatrixElement(
	const AbstractState &ket,
	const AbstractOperator &o
) const{
	//Only matrix elements between Sto3g states are supported.
	TBTKAssert(
		ket.getStateID() == StateID::STO3G,
		"Sto3g::getMatrixElement()",
		"Ket state with StateID '" << ket.getStateID() << "' is not"
		<< " supported.",
		""
	);

	//Confirm that the states either both have or both do not have spin
	//indices.
	TBTKAssert(
		(spinIndex == -1 && ((const Sto3g&)ket).spinIndex == -1)
		|| (spinIndex != -1 && ((const Sto3g&)ket).spinIndex != -1),
		"Sto3g::getMatrixElement()",
		"Bra and ket state must either both have or both not have a"
		<< " spin index.",
		""
	);

	switch(o.getOperatorID()){
	case AbstractOperator::OperatorID::Kinetic:
		return getKineticTerm(
			(const Sto3g&)ket,
			(const KineticOperator&)o
		);
	case AbstractOperator::OperatorID::NuclearPotential:
		return getNuclearPotentialTerm(
			(const Sto3g&)ket,
			(const NuclearPotentialOperator&)o
		);
	case AbstractOperator::OperatorID::HartreeFockPotential:
		return getHartreeFockPotentialTerm(
			(const Sto3g&)ket,
			(const HartreeFockPotentialOperator&)o
		);
	default:
		TBTKExit(
			"Sto3g::getMatrixElement()",
			"Unsupported OperatorID '" << o.getOperatorID()
			<< "'.",
			""
		);
	}
}

complex<double> Sto3g::getKineticTerm(
	const Sto3g &ket,
	const KineticOperator &o
) const{
	//Return zero if the states have unequal spin indices.
	if(spinIndex != -1)
		if(getIndex()[spinIndex] != ket.getIndex()[ket.spinIndex])
			return 0;

	//The implemented equations assumes that distances are given in terms
	//of the Bohr radius. Therefore distances have to be converted to this
	//unit.
	double bohrRadius = UnitHandler::getConstantInNaturalUnits("a_0");

	Vector3d braR = Vector3d(getCoordinates())/bohrRadius;
	Vector3d ketR = Vector3d(ket.getCoordinates())/bohrRadius;
	double deltaR = (braR - ketR).norm();

	//The main algorithm. See Appendix A in the book Modern Quantum
	//Chemistry: Introduction to Advanced Electronic Structure Theory by
	//A Szabo and NS Ostlund.
	double kineticTerm = 0;
	for(unsigned int n = 0; n < 3; n++){
		double alpha = gaussianExponents[n];
		double normA = pow(2*alpha/M_PI, 3/4.);
		double dA = contractionCoefficients[n];
		for(unsigned int np= 0; np < 3; np++){
			double beta = ket.gaussianExponents[np];
			double normB = pow(2*beta/M_PI, 3/4.);
			double dB = ket.contractionCoefficients[np];

			kineticTerm += dA*dB*normA*normB*(
				alpha*beta/(alpha + beta)
			)*(
				3 - 2*(
					alpha*beta/(alpha + beta)
				)*pow(deltaR, 2)
			)*pow(M_PI/(alpha + beta), 3/2.)*exp(
				-(alpha*beta/(alpha + beta))*pow(deltaR, 2)
			);
		}
	}

	//Multiply by the prefactor hbra^2/m. (1/2 is included in the
	//expression above)
	double hbar = UnitHandler::getConstantInNaturalUnits("hbar");
	kineticTerm *= pow(hbar, 2)/o.getMass();

	//Divide by the Bohr radius squared to account for the scale factor
	//comming from \nabla^2.
	kineticTerm /= pow(bohrRadius, 2);

	return kineticTerm;
}

complex<double> Sto3g::getNuclearPotentialTerm(
	const Sto3g &ket,
	const NuclearPotentialOperator &o
) const{
	//Return zero if the states have unequal spin indices.
	if(spinIndex != -1)
		if(getIndex()[spinIndex] != ket.getIndex()[ket.spinIndex])
			return 0;

	//The implemented equations assumes that distances are given in terms
	//of the Bohr radius. Therefore distances have to be converted to this
	//unit.
	double bohrRadius = UnitHandler::getConstantInNaturalUnits("a_0");

	Vector3d braR = Vector3d(getCoordinates())/bohrRadius;
	Vector3d ketR = Vector3d(ket.getCoordinates())/bohrRadius;
	double deltaR = (braR - ketR).norm();
	unsigned int Z = o.getNucleus().getAtomicNumber();
	Vector3d RC = o.getPosition()/bohrRadius;

	//The main algorithm. See Appendix A in the book Modern Quantum
	//Chemistry: Introduction to Advanced Electronic Structure Theory by
	//A Szabo and NS Ostlund.
	double nuclearPotentialTerm = 0;
	for(unsigned int n = 0; n < 3; n++){
		double alpha = gaussianExponents[n];
		double normA = pow(2*alpha/M_PI, 3/4.);
		double dA = contractionCoefficients[n];
		for(unsigned int np = 0; np < 3; np++){
			double beta = ket.gaussianExponents[np];
			double normB = pow(2*beta/M_PI, 3/4.);
			double dB = ket.contractionCoefficients[np];

			Vector3d RP = (alpha*braR + beta*ketR)/(alpha + beta);
			double deltaRPC = (RP - RC).norm();

			nuclearPotentialTerm -= dA*dB*normA*normB*(
				2*M_PI/(alpha + beta)
			)*Z*exp(
				-(alpha*beta/(alpha + beta))*pow(deltaR, 2)
			)*F_0((alpha + beta)*pow(deltaRPC, 2));
		}
	}

	//Multiply by the prefactor e^2/(4\pi\epsilon_0).
	double e = UnitHandler::getConstantInNaturalUnits("e");
	double epsilon_0 = UnitHandler::getConstantInNaturalUnits("epsilon_0");
	nuclearPotentialTerm *= pow(e, 2)/(4*M_PI*epsilon_0);

	//Divide by the Bohr radius to account for the scale factor comming
	//from 1/r.
	nuclearPotentialTerm /= bohrRadius;

	return nuclearPotentialTerm;
}

complex<double> Sto3g::getHartreeFockPotentialTerm(
	const Sto3g &ket,
	const HartreeFockPotentialOperator &o
) const{
	TBTKAssert(
		o.getFirstState().getStateID() == StateID::STO3G
		&& o.getSecondState().getStateID() == StateID::STO3G,
		"Sto3g::getHartreeFockPotentialTerm()",
		"The operator contains a state that does not have StateID"
		<< " 'StateID::STO3G', which is not supported.",
		""
	);

	complex<double> hartreeFockPotentialTerm
		= getSingleHartreeFockTerm(
			(const Sto3g&)o.getFirstState(),
			(const Sto3g&)o.getSecondState(),
			ket
		) - getSingleHartreeFockTerm(
			(const Sto3g&)o.getFirstState(),
			ket,
			(const Sto3g&)o.getSecondState()
		);

	//Multiply by the prefactor e^2/(4\pi\epsilon_0).
	double e = UnitHandler::getConstantInNaturalUnits("e");
	double epsilon_0 = UnitHandler::getConstantInNaturalUnits("epsilon_0");
	hartreeFockPotentialTerm *= pow(e, 2)/(4*M_PI*epsilon_0);

	//Divide by the Bohr radius to account for the scale factor comming
	//from 1/r.
	double bohrRadius = UnitHandler::getConstantInNaturalUnits("a_0");
	hartreeFockPotentialTerm /= bohrRadius;

	return hartreeFockPotentialTerm;
}

complex<double> Sto3g::getSingleHartreeFockTerm(
	const Sto3g &state1,
	const Sto3g &state2,
	const Sto3g &state3
) const{
	const Sto3g &state0 = *this;

	//Confirm that the states either all have or all do not have spin
	//indices.
	TBTKAssert(
		(
			state0.spinIndex == -1
			&& state1.spinIndex == -1
			&& state2.spinIndex == -1
			&& state3.spinIndex == -1
		) || (
			state0.spinIndex != -1
			&& state1.spinIndex != -1
			&& state2.spinIndex != -1
			&& state3.spinIndex != -1
		),
		"Sto3g::getSingleHartreeFockTerm()",
		"The states must either all have or all not have a spin"
		<< " index.",
		""
	);

	//Return zero if the spin indices does not match.
	if(state0.spinIndex != -1){
		if(
			state0.getIndex()[state0.spinIndex]
				!= state3.getIndex()[state3.spinIndex]
			|| state1.getIndex()[state1.spinIndex]
				!= state2.getIndex()[state2.spinIndex]
		){
			return 0;
		}
	}

	//The implemented equations assumes that distances are given in terms
	//of the Bohr radius. Therefore distances have to be converted to this
	//unit.
	double bohrRadius = UnitHandler::getConstantInNaturalUnits("a_0");

	Vector3d state0R = Vector3d(state0.getCoordinates())/bohrRadius;
	Vector3d state1R = Vector3d(state1.getCoordinates())/bohrRadius;
	Vector3d state2R = Vector3d(state2.getCoordinates())/bohrRadius;
	Vector3d state3R = Vector3d(state3.getCoordinates())/bohrRadius;
	double deltaR03 = (state0R - state3R).norm();
	double deltaR12 = (state1R - state2R).norm();

	//The main algorithm. See Appendix A in the book Modern Quantum
	//Chemistry: Introduction to Advanced Electronic Structure Theory by
	//A Szabo and NS Ostlund.
	//
	//Note: The ordering of the states are different from the ordering in
	//the integral expression in the Appendix. The states are here ordered
	//to appear on the form
	//Psi_0^{*}(x_1)Psi_1^{*}(x_2)Psi_2(x_2)Psi_3(x_1), while the
	//corresponding ordering in the book is
	//Psi_0^{*}(x_1)Psi_3(x_1)Psi_2^{*}(x_2)Psi_3(x_2).
	double result = 0;
	for(unsigned int n = 0; n < 3; n++){
		double alpha = state0.gaussianExponents[n];
		double norm0 = pow(2*alpha/M_PI, 3/4.);
		double d0 = state0.contractionCoefficients[n];
		for(unsigned int np = 0; np < 3; np++){
			double beta = state3.gaussianExponents[np];
			double norm3 = pow(2*beta/M_PI, 3/4.);
			double d3 = state3.contractionCoefficients[np];
			for(unsigned int npp = 0; npp < 3; npp++){
				double gamma = state1.gaussianExponents[npp];
				double norm1 = pow(2*gamma/M_PI, 3/4.);
				double d1 = state1.contractionCoefficients[npp];
				for(unsigned int nppp = 0; nppp < 3; nppp++){
					double delta
						= state2.gaussianExponents[nppp];
					double norm2 = pow(2*delta/M_PI, 3/4.);
					double d2
						= state2.contractionCoefficients[nppp];

					Vector3d pR = (
						alpha*state0R
						+ beta*state3R
					)/(alpha + beta);
					Vector3d qR = (
						gamma*state1R
						+ delta*state2R
					)/(gamma + delta);
					double deltaRPQ = (pR - qR).norm();

					result += d0*d1*d2*d3*norm0*norm1*norm2
						*norm3*(
							2*pow(M_PI, 5/2.)/(
								(alpha + beta)
								*(gamma + delta)
								*pow(
									alpha
									+ beta
									+ gamma
									+ delta,
									1/2.
								)
							)
						)*exp(
							-(
								alpha*beta/(
									alpha
									+ beta
								)
							)*pow(deltaR03, 2)
							- (
								gamma*delta/(
									gamma
									+ delta
								)
							)*pow(deltaR12, 2)
						)*F_0(
							(
								(alpha + beta)
								*(
									gamma
									+ delta
								)/(
									alpha
									+ beta
									+ gamma
									+ delta
								)
							)*pow(deltaRPQ, 2)
						);
				}
			}
		}
	}

	return result;
}

double Sto3g::F_0(double x){
	if(x != 0){
		return (1/2.)*sqrt(M_PI/x)*erf(sqrt(x));
	}
	else{
		//The limit of exp(sqrt(x))/sqrt(x) as x goes to zero.
		constexpr double LIMIT_VALUE = 1.1283791670578999;

		return (1/2.)*sqrt(M_PI)*LIMIT_VALUE;
	}
}

};	//End of namespace TBTK
