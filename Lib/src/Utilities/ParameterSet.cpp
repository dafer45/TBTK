/* Copyright 2016 Kristofer Björnson and Andreas Theiler
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

/** @file ParameterSet.cpp
 *
 *  @author Kristofer Björnson
 *  @author Andreas Theiler
 */

#include "TBTK/ParameterSet.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTKMacros.h"

using namespace std;

namespace TBTK{

ParameterSet::ParameterSet(){
}

ParameterSet::~ParameterSet(){
}

void ParameterSet::addInt(string name, int value){
	for(unsigned int n = 0; n < intParams.size(); n++){
		TBTKAssert(
			(get<0>(intParams.at(n))).compare(name) != 0,
			"Error in ParameterSet::addInt()",
			"Multiple definitions of parameter '" << name << "'.",
			""
		);
	}

	intParams.push_back(make_tuple(name, value));
}

void ParameterSet::addDouble(string name, double value){
	for(unsigned int n = 0; n < doubleParams.size(); n++){
		TBTKAssert(
			(get<0>(doubleParams.at(n))).compare(name) != 0,
			"ParameterSet::addDouble()",
			"Multiple definitions of parameter '" << name << "'.",
			""
		);
	}

	doubleParams.push_back(make_tuple(name, value));
}

void ParameterSet::addComplex(string name, complex<double> value){
	for(unsigned int n = 0; n < complexParams.size(); n++){
		TBTKAssert(
			(get<0>(complexParams.at(n))).compare(name) != 0,
			"ParameterSet::addComplex()",
			"Multiple definitions of parameter '" << name << "'.",
			""
		);
	}

	complexParams.push_back(make_tuple(name, value));
}

void ParameterSet::addString(string name, std::string value){
	for(unsigned int n = 0; n < stringParams.size(); n++){
		TBTKAssert(
			(get<0>(stringParams.at(n))).compare(name) != 0,
			"ParameterSet::addString()",
			"Multiple definitions of parameter '" << name << "'.",
			""
		);
	}

	stringParams.push_back(make_tuple(name, value));
}

void ParameterSet::addBool(string name, bool value){
	for(unsigned int n = 0; n < boolParams.size(); n++){
		TBTKAssert(
			(get<0>(boolParams.at(n))).compare(name) != 0,
			"ParameterSet::addBool()",
			"Multiple definitions of parameter '" << name << "'.",
			""
		);
	}

	boolParams.push_back(make_tuple(name, value));
}


void ParameterSet::setInt(std::string name, int value){
	for(unsigned int n = 0; n < intParams.size(); n++){
		if( (get<0>(intParams.at(n))).compare(name) == 0){
			get<1>(intParams.at(n)) = value;
			return;
		}
	}

	TBTKExit("Error in ParameterSet::setInt()",
		"Parameter '" << name << "' not found.",
		""
	);
}

void ParameterSet::setDouble(std::string name, double value){
	for(unsigned int n = 0; n < doubleParams.size(); n++){
		if( (get<0>(doubleParams.at(n))).compare(name) == 0){
			get<1>(doubleParams.at(n)) = value;
			return;
		}
	}

	TBTKExit("Error in ParameterSet::setDouble()",
		"Parameter '" << name << "' not found.",
		""
	);
}


void ParameterSet::setComplex(std::string name, std::complex<double> value){
	for(unsigned int n = 0; n < complexParams.size(); n++){
		if( (get<0>(complexParams.at(n))).compare(name) == 0){
			get<1>(complexParams.at(n)) = value;
			return;
		}
	}

	TBTKExit("Error in ParameterSet::setComplex()",
		"Parameter '" << name << "' not found.",
		""
	);
}

void ParameterSet::setString(std::string name, std::string value){
	for(unsigned int n = 0; n < stringParams.size(); n++){
		if( (get<0>(stringParams.at(n))).compare(name) == 0){
			get<1>(stringParams.at(n)) = value;
			return;
		}
	}

	TBTKExit("Error in ParameterSet::setString()",
		"Parameter '" << name << "' not found.",
		""
	);
}

void ParameterSet::setBool(std::string name, bool value){
	for(unsigned int n = 0; n < boolParams.size(); n++){
		if( (get<0>(boolParams.at(n))).compare(name) == 0){
			get<1>(boolParams.at(n)) = value;
			return;
		}
	}

	TBTKExit("Error in ParameterSet::setBool()",
		"Parameter '" << name << "' not found.",
		""
	);
}

int ParameterSet::getInt(string name) const {
	for(unsigned int n = 0; n < intParams.size(); n++){
		if((get<0>(intParams.at(n))).compare(name) == 0){
			return get<1>(intParams.at(n));
		}
	}

	TBTKExit(
		"ParameterSet::getInt()",
		"Parameter '" << name << "' not defined.",
		""
	);
}

double ParameterSet::getDouble(string name) const {
	for(unsigned int n = 0; n < doubleParams.size(); n++){
		if((get<0>(doubleParams.at(n))).compare(name) == 0){
			return get<1>(doubleParams.at(n));
		}
	}

	TBTKExit(
		"ParameterSet::getDouble()",
		"Parameter '" << name << "' not defined.",
		""
	);
}

complex<double> ParameterSet::getComplex(string name) const {
	for(unsigned int n = 0; n < complexParams.size(); n++){
		if((get<0>(complexParams.at(n))).compare(name) == 0){
			return get<1>(complexParams.at(n));
		}
	}

	TBTKExit(
		"ParameterSet::getComplex()",
		"Parameter '" << name << "' not defined.",
		""
	);
}

string ParameterSet::getString(string name) const {
	for(unsigned int n = 0; n < stringParams.size(); n++){
		if((get<0>(stringParams.at(n))).compare(name) == 0){
			return get<1>(stringParams.at(n));
		}
	}

	TBTKExit(
		"ParameterSet::getString()",
		"Parameter '" << name << "' not defined.",
		""
	);
}

bool ParameterSet::getBool(string name) const {
	for(unsigned int n = 0; n < boolParams.size(); n++){
		if((get<0>(boolParams.at(n))).compare(name) == 0){
			return get<1>(boolParams.at(n));
		}
	}

	TBTKExit(
		"ParameterSet::getBool()",
		"Parameter '" << name << "' not defined.",
		""
	);
}

int ParameterSet::getNumInt() const {
	return intParams.size();
}

int ParameterSet::getNumDouble() const {
	return doubleParams.size();
}

int ParameterSet::getNumComplex() const {
	return complexParams.size();
}

int ParameterSet::getNumString() const {
	return stringParams.size();
}

int ParameterSet::getNumBool() const {
	return boolParams.size();
}

std::string ParameterSet::getIntName(int n) const {
	return get<0>(intParams.at(n));
}

std::string ParameterSet::getDoubleName(int n) const {
	return get<0>(doubleParams.at(n));
}

std::string ParameterSet::getComplexName(int n) const {
	return get<0>(complexParams.at(n));
}

std::string ParameterSet::getStringName(int n) const {
	return get<0>(stringParams.at(n));
}

std::string ParameterSet::getBoolName(int n) const {
	return get<0>(boolParams.at(n));
}

int ParameterSet::getIntValue(int n) const {
	return get<1>(intParams.at(n));
}

double ParameterSet::getDoubleValue(int n) const {
	return get<1>(doubleParams.at(n));
}

complex<double> ParameterSet::getComplexValue(int n) const {
	return get<1>(complexParams.at(n));
}

string ParameterSet::getStringValue(int n) const {
	return get<1>(stringParams.at(n));
}

bool ParameterSet::getBoolValue(int n) const {
	return get<1>(boolParams.at(n));
}

bool ParameterSet::intExists(string name) const {
	for(unsigned int n = 0; n < intParams.size(); n++){
		if((get<0>(intParams.at(n))).compare(name) == 0){
			return true;
		}
	}

	return false;
}

bool ParameterSet::doubleExists(string name) const {
	for(unsigned int n = 0; n < doubleParams.size(); n++){
		if((get<0>(doubleParams.at(n))).compare(name) == 0){
			return true;
		}
	}

	return false;
}

bool ParameterSet::complexExists(string name) const {
	for(unsigned int n = 0; n < complexParams.size(); n++){
		if((get<0>(complexParams.at(n))).compare(name) == 0){
			return true;
		}
	}

	return false;
}

bool ParameterSet::stringExists(string name) const {
	for(unsigned int n = 0; n < stringParams.size(); n++){
		if((get<0>(stringParams.at(n))).compare(name) == 0){
			return true;
		}
	}

	return false;
}

bool ParameterSet::boolExists(string name) const {
	for(unsigned int n = 0; n < boolParams.size(); n++){
		if((get<0>(boolParams.at(n))).compare(name) == 0){
			return true;
		}
	}

	return false;
}

};	//End of namespace TBTK
