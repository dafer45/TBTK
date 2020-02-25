/* Copyright 2020 Kristofer Björnson
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
 *  @file Transport.h
 *  @brief Calculate transport properties.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_SOLVER_TRANSPORT
#define COM_DAFER45_TBTK_SOLVER_TRANSPORT

#include "TBTK/Communicator.h"
#include "TBTK/Property/GreensFunction.h"
#include "TBTK/Range.h"
#include "TBTK/Property/SelfEnergy.h"
#include "TBTK/Property/SpectralFunction.h"
#include "TBTK/Property/TransmissionRate.h"
#include "TBTK/Solver/Solver.h"

#include <complex>

namespace TBTK{
namespace Solver{

/** @brief Calculates transport properties. */
class Transport : public Solver, public Communicator{
public:
	/** Constructs a Solver::Transport. */
	Transport();

	/** Set the energy window. */
	void setEnergyWindow(
		double lowerBound,
		double upperBound,
		unsigned int resolution
	);

	/** Add lead. */
	void addLead(
		const Property::SelfEnergy &selfEnergy,
		double chemicalPotential,
		double temperature
	);

	/** Calculate transmission rate.
	 *
	 *  @return The transmission rate. */
	double calculateCurrent(
		unsigned int lead0/*,
		unsigned int lead1*/
	);
private:
	/** Green's function to use in calculations. */
	Property::GreensFunction greensFunction;

	/** Interacting Green's function. */
	Property::GreensFunction interactingGreensFunction;

	/** Spectral function. */
	Property::SpectralFunction spectralFunction;

	class Lead{
	public:
		Lead(
			const Property::SelfEnergy &selfEnergy,
			double chemicalPotential,
			double temperature
		);
		Property::SelfEnergy selfEnergy;
		Property::SelfEnergy broadening;
		Property::SelfEnergy inscattering;
		Property::EnergyResolvedProperty<double> energyResolvedCurrent;
		double current;
		double chemicalPotential;
		double temperature;
	};

	/** Leads. */
	std::vector<Lead> leads;

	/** Full self-energy. */
	Property::SelfEnergy fullSelfEnergy;

	/** Full inscattering. */
	Property::SelfEnergy fullInscattering;

	/** Correlation function. (See for example Eq. 11.1.5 in Quantum
	 *  Transport: Atom to Transistor by S. Datta, where the correlation
	 *  function also is denoted by the symbol G^{n}.) */
	Property::EnergyResolvedProperty<
		std::complex<double>
	> correlationFunction;

	/** Energy resolution. */
	Range energyRange;

	/** Calculate the Green's function. */
	void calculateGreensFunction();

	/** Calculate the interacting Green's function. */
	void calculateInteractingGreensFunction();

	/** Calculate the full self-energy. */
	void calculateFullSelfEnergy();

	/** Calculate the broadenings. */
	void calculateBroadenings();

	/** Calculate the inscatterings. */
	void calculateInscatterings();

	/** Calculate the full inscattering. */
	void calculateFullInscattering();

	/** Expand the self-energy to contain all the indices*/
	Property::SelfEnergy expandSelfEnergyIndexRange(
		const Property::SelfEnergy &selfEnergy
	) const;

	/** Calculate the correlation function. */
	void calculateCorrelationFunction();

	/** Calculate the spectra function. */
	void calculateSpectralFunction();

	/** Calculate the energy-resolved currents. */
	void calculateEnergyResolvedCurrents();

	/** Calculate currents. */
	void calculateCurrents();
};

inline void Transport::setEnergyWindow(
	double lowerBound,
	double upperBound,
	unsigned int resolution
){
	energyRange = Range(lowerBound, upperBound, resolution);
}

inline void Transport::addLead(
	const Property::SelfEnergy &selfEnergy,
	double chemicalPotential,
	double temperature
){
	leads.push_back(Lead(selfEnergy, chemicalPotential, temperature));
}

inline Transport::Lead::Lead(
	const Property::SelfEnergy &selfEnergy,
	double chemicalPotential,
	double temperature
){
	this->selfEnergy = selfEnergy;
	this->chemicalPotential = chemicalPotential;
	this->temperature = temperature;
}

};	//End of namespace Solver
};	//End of namespace TBTK

#endif
/// @endcond
