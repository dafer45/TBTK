/* Copyright 2018 Kristofer Björnson
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
 *  @file EnergyResolvedProperty.h
 *  @brief Base class for energy resolved Properties.
 *
 *  @author Kristofer Björnson
 */

#ifndef COM_DAFER45_TBTK_PROPERTY_ENERGY_RESOLVED_PROPERTY
#define COM_DAFER45_TBTK_PROPERTY_ENERGY_RESOLVED_PROPERTY

#include "TBTK/Property/AbstractProperty.h"
#include "TBTK/TBTKMacros.h"

#include <complex>
#include <vector>

namespace TBTK{
namespace Property{

/** @brief Base class for energy resolved Properties. */
template<typename DataType>
class EnergyResolvedProperty : public AbstractProperty<DataType>{
public:
	/** Enum class for specifying the energy type. */
	enum class EnergyType{Real, FermionicMatsubara, BosonicMatsubara};

	/** Constructs an uninitialized EnergyResolvedProperty. */
	EnergyResolvedProperty();

	/** Constructs an EnergyResolvedProperty with real energies on the
	 *  Custom format. [See AbstractProperty for detailed information about
	 *  the Custom format.]
	 *
	 *  @param indexTree IndexTree containing the @link Index Indices
	 *  @endlink for which the EnergyResolvedProperty should be contained.
	 *
	 *  @param lowerBound Lower bound for the energy.
	 *  @param upperBound Upper bound for the energy.
	 *  @param resolution Number of points to use for the energy. */
	EnergyResolvedProperty(
		const IndexTree &indexTree,
		double lowerBound,
		double upperBound,
		unsigned int resolution
	);

	/** Constructs an EnergyResolvedProperty with real energie on the
	 *  Custom format and initializes it with data. [See AbstractProperty
	 *  for detailed information about the Custom format and the raw data
	 *  format.]
	 *
	 *  @param indexTree IndexTree containing the @link Index Indices
	 *  @endlink for which the EnergyResolvedProperty should be contained.
	 *
	 *  @param lowerBound Lower bound for the energy.
	 *  @param upperBound Upper bound for the energy.
	 *  @param resolution Number of points to use for the energy.
	 *  @param data Raw data to initialize the EnergyResolvedProperty with.
	 */
	EnergyResolvedProperty(
		const IndexTree &indexTree,
		double lowerBound,
		double upperBound,
		unsigned int resolution,
		const DataType *data
	);

	/** Constructs an EnergyResolvedProperty with Matsubara energies
	 *  on the Custom format. [See AbstractProperty for detailed
	 *  information about the Custom format.]
	 *
	 *  @param indexTree IndexTree containing the @link Index Indices
	 *  @endlink for which the EnergyResolvedProperty should be contained.
	 *
	 *  @param lowerBound Lower bound for the energy.
	 *  @param upperBound Upper bound for the energy.
	 *  @param resolution Number of points to use for the energy. */
	EnergyResolvedProperty(
		EnergyType energyType,
		const IndexTree &indexTree,
		int lowerMatsubaraEnergyIndex,
		int upperMatsubaraEnergtIndex,
		double fundamentalMatsubaraEnergy
	);

	/** Constructs an EnergyResolvedProperty with Matsubara energies
	 *  on the Custom format and initializes it with data. [See
	 *  AbstractProperty for detailed information about the Custom format
	 *  and the raw data format.]
	 *
	 *  @param indexTree IndexTree containing the @link Index Indices
	 *  @endlink for which the EnergyResolvedProperty should be contained.
	 *
	 *  @param lowerBound Lower bound for the energy.
	 *  @param upperBound Upper bound for the energy.
	 *  @param resolution Number of points to use for the energy.
	 *  @param data Raw data to initialize the EnergyResolvedProperty with.
	 */
	EnergyResolvedProperty(
		EnergyType energyType,
		const IndexTree &indexTree,
		int lowerMatsubaraEnergyIndex,
		int upperMatsubaraEnergtIndex,
		double fundamentalMatsubaraEnergy,
		const DataType *data
	);

	/** Constructor. Constructs the EnergyResolvedProperty from a
	 *  serialization string.
	 *
	 *  @param serialization Serialization string from which to construct
	 *  the EnergyResolvedProperty. */
	EnergyResolvedProperty(
		const std::string &serialization,
		Serializable::Mode mode
	);

	/*** Get energy type.
	 *
	 *  @return The EnergyType. */
	EnergyType getEnergyType() const;

	/** Get lower bound for the energy.
	 *
	 *  @return Lower bound for the energy. */
	double getLowerBound() const;

	/** Get upper bound for the energy.
	 *
	 *  @return Upper bound for the energy. */
	double getUpperBound() const;

	/** Get the energy resolution (number of points used for the energy
	 *  axis).
	 *
	 *  @return The energy resolution. */
	unsigned int getResolution() const;

	/** Get the nth energy value.
	 *
	 *  @param n The energy index to get the energy for.
	 *
	 *  @return The energy for the nth energy index. */
	double getEnergy(unsigned int n) const;

	/** Get the lower Matsubara energy index. That is, l in the expression
	 *  E = (l + 2*n)*E_0.
	 *
	 *  @return The lowest Matsubara energy index. */
	int getLowerMatsubaraEnergyIndex() const;

	/** Get the upper Matsubara energy index. That is, l+N-1, where l is
	 *  the lowest Matsubara energy index and N is the number of Matsubara
	 *  energies.
	 *
	 *  @return The largest Matsubara energy index. */
	int getUpperMatsubaraEnergyIndex() const;

	/** Get the number of Matsubara energies.
	 *
	 *  @return The number of Matsubara energies. */
	unsigned int getNumMatsubaraEnergies() const;

	/** Get the fundamental Matsubara energy E_0 in the expression
	 *  E = (l + 2*n)*E_0. */
	double getFundamentalMatsubaraEnergy() const;

	/** Get the lower Matsubara energy.
	 *
	 *  @return The lowest Matsubara energy. */
	double getLowerMatsubaraEnergy() const;

	/** Get the upper Matsubara energy.
	 *
	 *  @return The highest Matsubara energy. */
	double getUpperMatsubaraEnergy() const;

	/** Get the nth Matsubara energy. */
	std::complex<double> getMatsubaraEnergy(unsigned int n) const;

	/** Overrides AbstractProperty::serialize(). */
	virtual std::string serialize(Serializable::Mode mode) const;
private:
	/** The energy type for the property. */
	EnergyType energyType;

	class RealEnergy{
	public:
		/** Lower bound for the energy. */
		double lowerBound;

		/** Upper bound for the energy. */
		double upperBound;

		/** Energy resolution. (Number of energy intervals) */
		unsigned int resolution;
	};

	class MatsubaraEnergy{
	public:
		/** The lowest Matsubara energy index l in the Expression . */
		int lowerMatsubaraEnergyIndex;

		/** The number of Matsubara energies. */
		int numMatsubaraEnergies;

		/** The energy E_0 in the expression E = (l + 2*n)*E_0. */
		double fundamentalMatsubaraEnergy;
	};

	/** Union of energy descriptors. */
	union EnergyDescriptor{
		RealEnergy realEnergy;
		MatsubaraEnergy matsubaraEnergy;
	};

	/** The actual energy descriptor. */
	EnergyDescriptor descriptor;
};

template<typename DataType>
EnergyResolvedProperty<DataType>::EnergyResolvedProperty(){
}

template<typename DataType>
EnergyResolvedProperty<DataType>::EnergyResolvedProperty(
	const IndexTree &indexTree,
	double lowerBound,
	double upperBound,
	unsigned int resolution
) :
	AbstractProperty<DataType>(indexTree, resolution)
{
	TBTKAssert(
		lowerBound <= upperBound,
		"EnergyResolvedProperty::EnergyResolvedProperty()",
		"Invalid energy bounds. The 'lowerBound=" << lowerBound << "'"
		" must be less or equal to the 'upperBound=" << upperBound
		<< "'.",
		""
	);
	TBTKAssert(
		resolution > 0,
		"EnergyResolvedProperty::EnergyResolvedProperty()",
		"The 'resolution' must be larger than 0.",
		""
	);

	energyType = EnergyType::Real;
	descriptor.realEnergy.lowerBound = lowerBound;
	descriptor.realEnergy.upperBound = upperBound;
	descriptor.realEnergy.resolution = resolution;
}

template<typename DataType>
EnergyResolvedProperty<DataType>::EnergyResolvedProperty(
	const IndexTree &indexTree,
	double lowerBound,
	double upperBound,
	unsigned int resolution,
	const DataType *data
) :
	AbstractProperty<DataType>(indexTree, resolution, data)
{
	TBTKAssert(
		lowerBound < upperBound,
		"EnergyResolvedProperty::EnergyResolvedProperty()",
		"Invalid energy bounds. The 'lowerBound=" << lowerBound << "'"
		" must be smaller than the 'upperBound=" << upperBound << "'.",
		""
	);
	TBTKAssert(
		resolution > 0,
		"EnergyResolvedProperty::EnergyResolvedProperty()",
		"The 'resolution' must be larger than 0.",
		""
	);

	energyType = EnergyType::Real;
	descriptor.realEnergy.lowerBound = lowerBound;
	descriptor.realEnergy.upperBound = upperBound;
	descriptor.realEnergy.resolution = resolution;
}

template<typename DataType>
EnergyResolvedProperty<DataType>::EnergyResolvedProperty(
	EnergyType energyType,
	const IndexTree &indexTree,
	int lowerMatsubaraEnergyIndex,
	int upperMatsubaraEnergyIndex,
	double fundamentalMatsubaraEnergy
) :
	AbstractProperty<DataType>(
		indexTree,
		(upperMatsubaraEnergyIndex-lowerMatsubaraEnergyIndex)/2 + 1
	)
{
	TBTKAssert(
		lowerMatsubaraEnergyIndex <= upperMatsubaraEnergyIndex,
		"EnergyResolvedProperty::EnergyResolvedProperty()",
		"Invalid Matsubara energy bounds. The"
		" 'lowerMatsubaraEnergyIndex=" << lowerMatsubaraEnergyIndex
		<< "' must be less or equal to the 'upperMatsubaraEnergyIndex="
		<< upperMatsubaraEnergyIndex << "'.",
		""
	);
	TBTKAssert(
		fundamentalMatsubaraEnergy > 0,
		"EnergyResolvedProperty::energyResolvedProperty()",
		"The 'fundamentalMatsubaraEnergy' must be larger than 0.",
		""
	);

	switch(energyType){
	case EnergyType::FermionicMatsubara:
		TBTKAssert(
			abs(lowerMatsubaraEnergyIndex%2) == 1,
			"EnergyResolvedProperty::EnergyResolvedProperty()",
			"The 'lowerMatsubaraEnergyIndex="
			<< lowerMatsubaraEnergyIndex << "' must be odd for"
			<< " EnergyType::FermionicMatsubara.",
			""
		);
		TBTKAssert(
			abs(upperMatsubaraEnergyIndex%2) == 1,
			"EnergyResolvedProperty::EnergyResolvedProperty()",
			"The 'upperMatsubaraEnergyIndex="
			<< upperMatsubaraEnergyIndex << "' must be odd for"
			<< " EnergyType::FermionicMatsubara.",
			""
		);

		this->energyType = energyType;
		descriptor.matsubaraEnergy.lowerMatsubaraEnergyIndex
			= lowerMatsubaraEnergyIndex;
		descriptor.matsubaraEnergy.numMatsubaraEnergies	= (
			upperMatsubaraEnergyIndex-lowerMatsubaraEnergyIndex)/2
			+ 1;
		descriptor.matsubaraEnergy.fundamentalMatsubaraEnergy
			= fundamentalMatsubaraEnergy;

		break;
	case EnergyType::BosonicMatsubara:
		TBTKAssert(
			lowerMatsubaraEnergyIndex%2 == 0,
			"EnergyResolvedProperty::EnergyResolvedProperty()",
			"The 'lowerMatsubaraEnergyIndex="
			<< lowerMatsubaraEnergyIndex << "' must be even for"
			<< " EnergyType::BosonicMatsubara.",
			""
		);
		TBTKAssert(
			upperMatsubaraEnergyIndex%2 == 0,
			"EnergyResolvedProperty::EnergyResolvedProperty()",
			"The 'upperMatsubaraEnergyIndex="
			<< upperMatsubaraEnergyIndex << "' must be even for"
			<< " EnergyType::BosonicMatsubara.",
			""
		);

		this->energyType = energyType;
		descriptor.matsubaraEnergy.lowerMatsubaraEnergyIndex
			= lowerMatsubaraEnergyIndex;
		descriptor.matsubaraEnergy.numMatsubaraEnergies	= (
			upperMatsubaraEnergyIndex-lowerMatsubaraEnergyIndex)/2
			+ 1;
		descriptor.matsubaraEnergy.fundamentalMatsubaraEnergy
			= fundamentalMatsubaraEnergy;

		break;
	default:
		TBTKExit(
			"EnergyResolvedProperty::EnergyResolvedProperty()",
			"The 'energyType' must be"
			" EnergyType::FermionicMatsubara or"
			" EnergyType::BosonicMatsubara.",
			""
		);
	}
}

template<typename DataType>
EnergyResolvedProperty<DataType>::EnergyResolvedProperty(
	EnergyType energyType,
	const IndexTree &indexTree,
	int lowerMatsubaraEnergyIndex,
	int upperMatsubaraEnergyIndex,
	double fundamentalMatsubaraEnergy,
	const DataType *data
) :
	AbstractProperty<DataType>(
		indexTree,
		(upperMatsubaraEnergyIndex-lowerMatsubaraEnergyIndex)/2 + 1,
		data
	)
{
	TBTKAssert(
		lowerMatsubaraEnergyIndex <= upperMatsubaraEnergyIndex,
		"EnergyResolvedProperty::EnergyResolvedProperty()",
		"Invalid Matsubara energy bounds. The"
		" 'lowerMatsubaraEnergyIndex=" << lowerMatsubaraEnergyIndex
		<< "' must be less or equal to the 'upperMatsubaraEnergyIndex="
		<< upperMatsubaraEnergyIndex << "'.",
		""
	);
	TBTKAssert(
		fundamentalMatsubaraEnergy > 0,
		"EnergyResolvedProperty::energyResolvedProperty()",
		"The 'fundamentalMatsubaraEnergy' must be larger than 0.",
		""
	);

	switch(energyType){
	case EnergyType::FermionicMatsubara:
		TBTKAssert(
			abs(lowerMatsubaraEnergyIndex%2) == 1,
			"EnergyResolvedProperty::EnergyResolvedProperty()",
			"The 'lowerMatsubaraEnergyIndex="
			<< lowerMatsubaraEnergyIndex << "' must be odd for"
			<< " EnergyType::FermionicMatsubara.",
			""
		);
		TBTKAssert(
			abs(upperMatsubaraEnergyIndex%2) == 1,
			"EnergyResolvedProperty::EnergyResolvedProperty()",
			"The 'uppererMatsubaraEnergyIndex="
			<< upperMatsubaraEnergyIndex << "' must be odd for"
			<< " EnergyType::FermionicMatsubara.",
			""
		);

		this->energyType = energyType;
		descriptor.matsubaraEnergy.lowerMatsubaraEnergyIndex
			= lowerMatsubaraEnergyIndex;
		descriptor.matsubaraEnergy.numMatsubaraEnergies	= (
			upperMatsubaraEnergyIndex-lowerMatsubaraEnergyIndex)/2
			+ 1;
		descriptor.matsubaraEnergy.fundamentalMatsubaraEnergy
			= fundamentalMatsubaraEnergy;

		break;
	case EnergyType::BosonicMatsubara:
		TBTKAssert(
			lowerMatsubaraEnergyIndex%2 == 0,
			"EnergyResolvedProperty::EnergyResolvedProperty()",
			"The 'lowerMatsubaraEnergyIndex="
			<< lowerMatsubaraEnergyIndex << "' must be even for"
			<< " EnergyType::BosonicMatsubara.",
			""
		);
		TBTKAssert(
			upperMatsubaraEnergyIndex%2 == 0,
			"EnergyResolvedProperty::EnergyResolvedProperty()",
			"The 'uppererMatsubaraEnergyIndex="
			<< upperMatsubaraEnergyIndex << "' must be even for"
			<< " EnergyType::BosonicMatsubara.",
			""
		);

		this->energyType = energyType;
		descriptor.matsubaraEnergy.lowerMatsubaraEnergyIndex
			= lowerMatsubaraEnergyIndex;
		descriptor.matsubaraEnergy.numMatsubaraEnergies	= (
			upperMatsubaraEnergyIndex-lowerMatsubaraEnergyIndex)/2
			+ 1;
		descriptor.matsubaraEnergy.fundamentalMatsubaraEnergy
			= fundamentalMatsubaraEnergy;

		break;
	default:
		TBTKExit(
			"EnergyResolvedProperty::EnergyResolvedProperty()",
			"The 'energyType' must be"
			" EnergyType::FermionicMatsubara or"
			" EnergyType::BosonicMatsubara.",
			""
		);
	}
}

template<typename DataType>
EnergyResolvedProperty<DataType>::EnergyResolvedProperty(
	const std::string &serialization,
	Serializable::Mode mode
) :
	AbstractProperty<DataType>(
		Serializable::extract(
			serialization,
			mode,
			"abstractProperty"
		),
		mode
	)
{
	TBTKAssert(
		Serializable::validate(serialization, "EnergyResolvedProperty", mode),
		"Property::EnergyResolvedProperty::EnergyResolvedProperty()",
		"Unable to parse string as EnergyResolvedProperty '"
		<< serialization << "'.",
		""
	);

	switch(mode){
	case Serializable::Mode::JSON:
		try{
			nlohmann::json j = nlohmann::json::parse(serialization);
			std::string et = j.at("energyType").get<std::string>();
			if(et.compare("Real") == 0){
				energyType = EnergyType::Real;
				descriptor.realEnergy.lowerBound
					= j.at("lowerBound").get<double>();
				descriptor.realEnergy.upperBound
					= j.at("upperBound").get<double>();
				descriptor.realEnergy.resolution
					= j.at("resolution").get<double>();
			}
			else if(et.compare("FermionicMatsubara") == 0){
				energyType = EnergyType::FermionicMatsubara;
				descriptor.matsubaraEnergy.lowerMatsubaraEnergyIndex
					= j.at("lowerMatsubaraEnergyIndex");
				descriptor.matsubaraEnergy.numMatsubaraEnergies
					= j.at("numMatsubaraEnergies");
				descriptor.matsubaraEnergy.fundamentalMatsubaraEnergy
					= j.at("fundamentalMatsubaraEnergy");
			}
			else if(et.compare("BosonicMatsubara") == 0){
				energyType = EnergyType::BosonicMatsubara;
				descriptor.matsubaraEnergy.lowerMatsubaraEnergyIndex
					= j.at("lowerMatsubaraEnergyIndex");
				descriptor.matsubaraEnergy.numMatsubaraEnergies
					= j.at("numMatsubaraEnergies");
				descriptor.matsubaraEnergy.fundamentalMatsubaraEnergy
					= j.at("fundamentalMatsubaraEnergy");
			}
		}
		catch(nlohmann::json::exception e){
			TBTKExit(
				"Proerty::EnergyResolvedProperty::EnergyResolvedProperty()",
				"Unable to parse string as"
				<< " EnergyResolvedProperty '" << serialization
				<< "'.",
				""
			);
		}

		break;
	default:
		TBTKExit(
			"Property::EnergyResolvedProperty::EnergyResolvedProperty()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

template<typename DataType>
inline typename EnergyResolvedProperty<DataType>::EnergyType
EnergyResolvedProperty<DataType>::getEnergyType() const{
	return energyType;
}

template<typename DataType>
inline double EnergyResolvedProperty<DataType>::getLowerBound() const{
	TBTKAssert(
		energyType == EnergyType::Real,
		"GreensFunction::getLowerBound()",
		"The Property is not of the type EnergyType::Real.",
		""
	);

	return descriptor.realEnergy.lowerBound;
}

template<typename DataType>
inline double EnergyResolvedProperty<DataType>::getUpperBound() const{
	TBTKAssert(
		energyType == EnergyType::Real,
		"GreensFunction::getUpperBound()",
		"The Property is not of the type EnergyType::Real.",
		""
	);

	return descriptor.realEnergy.upperBound;
}

template<typename DataType>
inline unsigned int EnergyResolvedProperty<DataType>::getResolution() const{
	TBTKAssert(
		energyType == EnergyType::Real,
		"GreensFunction::getResolution()",
		"The Property is not of the type EnergyType::Real.",
		""
	);

	return descriptor.realEnergy.resolution;
}

template<typename DataType>
inline double EnergyResolvedProperty<DataType>::getEnergy(
	unsigned int n
) const{
	TBTKAssert(
		energyType == EnergyType::Real,
		"GreensFunction::getEnergy()",
		"The Property is not of the type EnergyType::Real.",
		""
	);

	double dE;
	if(descriptor.realEnergy.resolution == 1)
		dE = 0;
	else
		dE = (
			descriptor.realEnergy.upperBound
			- descriptor.realEnergy.lowerBound
		)/(descriptor.realEnergy.resolution - 1);

	return descriptor.realEnergy.lowerBound + ((int)n)*dE;
}

template<typename DataType>
inline int EnergyResolvedProperty<DataType>::getLowerMatsubaraEnergyIndex(
) const{
	TBTKAssert(
		energyType == EnergyType::FermionicMatsubara
		|| energyType == EnergyType::BosonicMatsubara,
		"GreensFunction::getLowerMatsubaraEnergyIndex()",
		"The Property is not of the type"
		<< " EnergyType::FermionicMatsubara or"
		<< " EnergyType::BosonicMatsubara.",
		""
	);

	return descriptor.matsubaraEnergy.lowerMatsubaraEnergyIndex;
}

template<typename DataType>
inline int EnergyResolvedProperty<DataType>::getUpperMatsubaraEnergyIndex(
) const{
	TBTKAssert(
		energyType == EnergyType::FermionicMatsubara
		|| energyType == EnergyType::BosonicMatsubara,
		"GreensFunction::getUpperMatsubaraEnergyIndex()",
		"The Property is not of the type"
		<< " EnergyType::FermionicMatsubara or"
		<< " EnergyType::BosonicMatsubara.",
		""
	);

	return descriptor.matsubaraEnergy.lowerMatsubaraEnergyIndex
	 + 2*(descriptor.matsubaraEnergy.numMatsubaraEnergies - 1);
}

template<typename DataType>
inline unsigned int EnergyResolvedProperty<DataType>::getNumMatsubaraEnergies() const{
	TBTKAssert(
		energyType == EnergyType::FermionicMatsubara
		|| energyType == EnergyType::BosonicMatsubara,
		"GreensFunction::getNumMatsubaraEnergies()",
		"The Property is not of the type"
		<< " EnergyType::FermionicMatsubara or"
		<< " EnergyType::BosonicMatsubara.",
		""
	);

	return descriptor.matsubaraEnergy.numMatsubaraEnergies;
}

template<typename DataType>
inline double EnergyResolvedProperty<
	DataType
>::getFundamentalMatsubaraEnergy() const{
	TBTKAssert(
		energyType == EnergyType::FermionicMatsubara
		|| energyType == EnergyType::BosonicMatsubara,
		"GreensFunction::getFundamentalMatsubaraEnergy()",
		"The Property is not of the type"
		<< " EnergyType::FermionicMatsubara or"
		<< " EnergyType::BosonicMatsubara.",
		""
	);

	return descriptor.matsubaraEnergy.fundamentalMatsubaraEnergy;
}

template<typename DataType>
inline double EnergyResolvedProperty<DataType>::getLowerMatsubaraEnergy(
) const{
	TBTKAssert(
		energyType == EnergyType::FermionicMatsubara
		|| energyType == EnergyType::BosonicMatsubara,
		"GreensFunction::getLowerMatsubaraEnergy()",
		"The Property is not of the type"
		<< " EnergyType::FermionicMatsubara or"
		<< " EnergyType::BosonicMatsubara.",
		""
	);

	return descriptor.matsubaraEnergy.lowerMatsubaraEnergyIndex
		*descriptor.matsubaraEnergy.fundamentalMatsubaraEnergy;
}

template<typename DataType>
inline double EnergyResolvedProperty<DataType>::getUpperMatsubaraEnergy(
) const{
	TBTKAssert(
		energyType == EnergyType::FermionicMatsubara
		|| energyType == EnergyType::BosonicMatsubara,
		"GreensFunction::getUpperMatsubaraEnergyIndex()",
		"The Property is not of the type"
		<< " EnergyType::FermionicMatsubara or"
		<< " EnergyType::BosonicMatsubara.",
		""
	);

	return (
			descriptor.matsubaraEnergy.lowerMatsubaraEnergyIndex
			+ 2*(descriptor.matsubaraEnergy.numMatsubaraEnergies-1)
		)*descriptor.matsubaraEnergy.fundamentalMatsubaraEnergy;
}

template<typename DataType>
inline std::complex<double> EnergyResolvedProperty<
	DataType
>::getMatsubaraEnergy(
	unsigned int n
) const{
	TBTKAssert(
		energyType == EnergyType::FermionicMatsubara
		|| energyType == EnergyType::BosonicMatsubara,
		"GreensFunction::getMatsubaraEnergy()",
		"The Property is not of the type"
		<< " EnergyType::FermionicMatsubara or"
		<< " EnergyType::BosonicMatsubara.",
		""
	);

	return std::complex<double>(
		0,
		(descriptor.matsubaraEnergy.lowerMatsubaraEnergyIndex + 2*(int)n)
		*descriptor.matsubaraEnergy.fundamentalMatsubaraEnergy
	);
}

template<typename DataType>
inline std::string EnergyResolvedProperty<DataType>::serialize(Serializable::Mode mode) const{
	switch(mode){
	case Serializable::Mode::JSON:
	{
		nlohmann::json j;
		j["id"] = "EnergyResolvedProperty";
		switch(energyType){
		case EnergyType::Real:
			j["energyType"] = "Real";
			j["lowerBound"] = descriptor.realEnergy.lowerBound;
			j["upperBound"] = descriptor.realEnergy.upperBound;
			j["resolution"] = descriptor.realEnergy.resolution;

			break;
		case EnergyType::FermionicMatsubara:
			j["energyType"] = "FermionicMatsubara";
			j["lowerMatsubaraEnergyIndex"]
				= descriptor.matsubaraEnergy.lowerMatsubaraEnergyIndex;
			j["numMatsubaraEnergies"]
				= descriptor.matsubaraEnergy.numMatsubaraEnergies;
			j["fundamentalMatsubaraEnergy"]
				= descriptor.matsubaraEnergy.fundamentalMatsubaraEnergy;

			break;
		case EnergyType::BosonicMatsubara:
			j["energyType"] = "BosonicMatsubara";
			j["lowerMatsubaraEnergyIndex"]
				= descriptor.matsubaraEnergy.lowerMatsubaraEnergyIndex;
			j["numMatsubaraEnergies"]
				= descriptor.matsubaraEnergy.numMatsubaraEnergies;
			j["fundamentalMatsubaraEnergy"]
				= descriptor.matsubaraEnergy.fundamentalMatsubaraEnergy;

			break;
		default:
			TBTKExit(
				"Property::EnergyResolvedProperty::serialize()",
				"Unknown EnergyType.",
				"This should never happen, contact the developer."
			);
		}
		j["abstractProperty"] = nlohmann::json::parse(
			AbstractProperty<DataType>::serialize(mode)
		);

		return j.dump();
	}
	default:
		TBTKExit(
			"Property::EnergyResolvedProperty::serialize()",
			"Only Serializable::Mode::JSON is supported yet.",
			""
		);
	}
}

};	//End namespace Property
};	//End namespace TBTK

#endif
