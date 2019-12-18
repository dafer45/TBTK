/* Copyright 2016 Kristofer Björnson
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

/** @file UnitHandler.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Quantity/Constants.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTKMacros.h"
#include "TBTK/UnitHandler.h"

#include <iostream>
#include <sstream>

#ifdef M_E	//Avoid name clash with math.h macro M_E
	#define M_E_temp M_E
	#undef M_E
#endif

using namespace std;

namespace TBTK{

map<
	string,
	Quantity::Constant
> UnitHandler::constantsDefaultUnits;
map<string, double> UnitHandler::constantsBaseUnits;

tuple<
	Quantity::Angle::Unit,
	Quantity::Charge::Unit,
	Quantity::Count::Unit,
	Quantity::Energy::Unit,
	Quantity::Length::Unit,
	Quantity::Temperature::Unit,
	Quantity::Time::Unit
> UnitHandler::units = make_tuple(
	Quantity::Angle::Unit::rad,
	Quantity::Charge::Unit::C,
	Quantity::Count::Unit::pcs,
	Quantity::Energy::Unit::eV,
	Quantity::Length::Unit::m,
	Quantity::Temperature::Unit::K,
	Quantity::Time::Unit::s
);

tuple<
	double,
	double,
	double,
	double,
	double,
	double,
	double
> UnitHandler::scales = make_tuple(1, 1, 1, 1, 1, 1, 1);

double UnitHandler::getConstantInBaseUnits(const std::string &name){
	return constantsBaseUnits.at(name);
}

double UnitHandler::getConstantInNaturalUnits(const std::string &name){
	double value = getConstantInBaseUnits(name);

	Quantity::Constant constant = constantsDefaultUnits.at(name);
	int angleExponent = constant.getExponent<Quantity::Angle>();
	int chargeExponent = constant.getExponent<Quantity::Charge>();
	int countExponent = constant.getExponent<Quantity::Count>();
	int energyExponent = constant.getExponent<Quantity::Energy>();
	int lengthExponent = constant.getExponent<Quantity::Length>();
	int temperatureExponent = constant.getExponent<Quantity::Temperature>();
	int timeExponent = constant.getExponent<Quantity::Time>();
	for(int n = 0; n < angleExponent; n++)
		value /= getScale<Quantity::Angle>();
	for(int n = 0; n < -angleExponent; n++)
		value *= getScale<Quantity::Angle>();
	for(int n = 0; n < chargeExponent; n++)
		value /= getScale<Quantity::Charge>();
	for(int n = 0; n < -chargeExponent; n++)
		value *= getScale<Quantity::Charge>();
	for(int n = 0; n < countExponent; n++)
		value /= getScale<Quantity::Count>();
	for(int n = 0; n < -countExponent; n++)
		value *= getScale<Quantity::Count>();
	for(int n = 0; n < energyExponent; n++)
		value /= getScale<Quantity::Energy>();
	for(int n = 0; n < -energyExponent; n++)
		value *= getScale<Quantity::Energy>();
	for(int n = 0; n < lengthExponent; n++)
		value /= getScale<Quantity::Length>();
	for(int n = 0; n < -lengthExponent; n++)
		value *= getScale<Quantity::Length>();
	for(int n = 0; n < temperatureExponent; n++)
		value /= getScale<Quantity::Temperature>();
	for(int n = 0; n < -temperatureExponent; n++)
		value *= getScale<Quantity::Temperature>();
	for(int n = 0; n < timeExponent; n++)
		value /= getScale<Quantity::Time>();
	for(int n = 0; n < -timeExponent; n++)
		value *= getScale<Quantity::Time>();

	return value;
}

string UnitHandler::getUnitString(const std::string &constantName){
	try{
		Quantity::Constant constant
			= constantsDefaultUnits.at(constantName);
		return getUnitString(
			constant.getExponent<Quantity::Angle>(),
			constant.getExponent<Quantity::Charge>(),
			constant.getExponent<Quantity::Count>(),
			constant.getExponent<Quantity::Energy>(),
			constant.getExponent<Quantity::Length>(),
			constant.getExponent<Quantity::Temperature>(),
			constant.getExponent<Quantity::Time>()
		);
	}
	catch(const out_of_range &e){
		TBTKExit(
			"UnitHandler::getUnitString()",
			"Unknown constant '" << constantName << "'.",
			""
		);
	}
}

void UnitHandler::updateConstants(){
	constantsBaseUnits.clear();
	for(auto c : constantsDefaultUnits){
		const string &name = c.first;
		Quantity::Constant constant = c.second;
		double value = constant;
		int angleExponent = constant.getExponent<Quantity::Angle>();
		int chargeExponent = constant.getExponent<Quantity::Charge>();
		int countExponent = constant.getExponent<Quantity::Count>();
		int energyExponent = constant.getExponent<Quantity::Energy>();
		int lengthExponent = constant.getExponent<Quantity::Length>();
		int temperatureExponent
			= constant.getExponent<Quantity::Temperature>();
		int timeExponent = constant.getExponent<Quantity::Time>();
		for(int n = 0; n < angleExponent; n++)
			value *= getConversionFactor<Quantity::Angle>();
		for(int n = 0; n < -angleExponent; n++)
			value /= getConversionFactor<Quantity::Angle>();
		for(int n = 0; n < chargeExponent; n++)
			value *= getConversionFactor<Quantity::Charge>();
		for(int n = 0; n < -chargeExponent; n++)
			value /= getConversionFactor<Quantity::Charge>();
		for(int n = 0; n < countExponent; n++)
			value *= getConversionFactor<Quantity::Count>();
		for(int n = 0; n < -countExponent; n++)
			value /= getConversionFactor<Quantity::Count>();
		for(int n = 0; n < energyExponent; n++)
			value *= getConversionFactor<Quantity::Energy>();
		for(int n = 0; n < -energyExponent; n++)
			value /= getConversionFactor<Quantity::Energy>();
		for(int n = 0; n < lengthExponent; n++)
			value *= getConversionFactor<Quantity::Length>();
		for(int n = 0; n < -lengthExponent; n++)
			value /= getConversionFactor<Quantity::Length>();
		for(int n = 0; n < temperatureExponent; n++)
			value *= getConversionFactor<Quantity::Temperature>();
		for(int n = 0; n < -temperatureExponent; n++)
			value /= getConversionFactor<Quantity::Temperature>();
		for(int n = 0; n < timeExponent; n++)
			value *= getConversionFactor<Quantity::Time>();
		for(int n = 0; n < -timeExponent; n++)
			value /= getConversionFactor<Quantity::Time>();

		constantsBaseUnits[name] = value;
	}
}

void UnitHandler::initialize(){
	constantsDefaultUnits["e"] = Quantity::Constants::get("e");
	constantsDefaultUnits["c"] = Quantity::Constants::get("c");
	constantsDefaultUnits["N_A"] = Quantity::Constants::get("N_A");
	constantsDefaultUnits["a_0"] = Quantity::Constants::get("a_0");
	constantsDefaultUnits["h"] = Quantity::Constants::get("h");
	constantsDefaultUnits["k_B"] = Quantity::Constants::get("k_B");
	constantsDefaultUnits["m_e"] = Quantity::Constants::get("m_e");
	constantsDefaultUnits["m_p"] = Quantity::Constants::get("m_p");
	constantsDefaultUnits["mu_0"] = Quantity::Constants::get("mu_0");
	constantsDefaultUnits["epsilon_0"]
		= Quantity::Constants::get("epsilon_0");
	constantsDefaultUnits["hbar"] = Quantity::Constants::get("hbar");
	constantsDefaultUnits["mu_B"] = Quantity::Constants::get("mu_B");
	constantsDefaultUnits["mu_N"] = Quantity::Constants::get("mu_N");

	updateConstants();
}

};

#ifdef M_E_temp
	#define M_E M_E_temp
	#undef M_E_temp
#endif
