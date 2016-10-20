/** @file ParameterSet.cpp
 *
 *  @author Kristofer Björnson
 *  @author Andreas Theiler
 */

#include "../include/ParameterSet.h"

#include <iostream>

using namespace std;

namespace TBTK{
namespace Util{

ParameterSet::ParameterSet(){
}

ParameterSet::~ParameterSet(){
}

void ParameterSet::addInt(std::string name, int value){
	for(unsigned int n = 0; n < intParams.size(); n++){
		if((get<0>(intParams.at(n))).compare(name) == 0){
			cout << "Error in ParameterSet::addInt(): Multiple definition of parameter '" << name << "'.\n";
			exit(1);
		}
	}

	intParams.push_back(make_tuple(name, value));
}

void ParameterSet::addDouble(std::string name, double value){
	for(unsigned int n = 0; n < doubleParams.size(); n++){
		if((get<0>(doubleParams.at(n))).compare(name) == 0){
			cout << "Error in ParameterSet::addDouble(): Multiple definition of parameter '" << name << "'.\n";
			exit(1);
		}
	}

	doubleParams.push_back(make_tuple(name, value));
}

void ParameterSet::addComplex(std::string name, complex<double> value){
	for(unsigned int n = 0; n < complexParams.size(); n++){
		if((get<0>(complexParams.at(n))).compare(name) == 0){
			cout << "Error in ParameterSet::addComplex(): Multiple definition of parameter '" << name << "'.\n";
			exit(1);
		}
	}

	complexParams.push_back(make_tuple(name, value));
}

void ParameterSet::addString(std::string name, std::string value){
	for(unsigned int n = 0; n < stringParams.size(); n++){
		if((get<0>(stringParams.at(n))).compare(name) == 0){
			cout << "Error in ParameterSet::addString(): Multiple definition of parameter '" << name << "'.\n";
			exit(1);
		}
	}

	stringParams.push_back(make_tuple(name, value));
}

void ParameterSet::addBool(std::string name, bool value){
	for(unsigned int n = 0; n < boolParams.size(); n++){
		if((get<0>(boolParams.at(n))).compare(name) == 0){
			cout << "Error in ParameterSet::addBool(): Multiple definition of parameter '" << name << "'.\n";
			exit(1);
		}
	}

	boolParams.push_back(make_tuple(name, value));
}

int ParameterSet::getInt(string name) const {
	for(unsigned int n = 0; n < intParams.size(); n++){
		if((get<0>(intParams.at(n))).compare(name) == 0){
			return get<1>(intParams.at(n));
		}
	}

	cout << "Error in ParameterSet::getInt(): Parameter '" << name << "' not defined.\n";
	exit(1);
}

double ParameterSet::getDouble(string name) const {
	for(unsigned int n = 0; n < doubleParams.size(); n++){
		if((get<0>(doubleParams.at(n))).compare(name) == 0){
			return get<1>(doubleParams.at(n));
		}
	}

	cout << "Error in ParameterSet::getDouble(): Parameter '" << name << "' not defined.\n";
	exit(1);
}

complex<double> ParameterSet::getComplex(string name) const {
	for(unsigned int n = 0; n < complexParams.size(); n++){
		if((get<0>(complexParams.at(n))).compare(name) == 0){
			return get<1>(complexParams.at(n));
		}
	}

	cout << "Error in ParameterSet::getComplex(): Parameter '" << name << "' not defined.\n";
	exit(1);
}

string ParameterSet::getString(string name) const {
	for(unsigned int n = 0; n < stringParams.size(); n++){
		if((get<0>(stringParams.at(n))).compare(name) == 0){
			return get<1>(stringParams.at(n));
		}
	}

	cout << "Error in ParameterSet::getString(): Parameter '" << name << "' not defined.\n";
	exit(1);
}

bool ParameterSet::getBool(string name) const {
	for(unsigned int n = 0; n < boolParams.size(); n++){
		if((get<0>(boolParams.at(n))).compare(name) == 0){
			return get<1>(boolParams.at(n));
		}
	}

	cout << "Error in ParameterSet::getBool(): Parameter '" << name << "' not defined.\n";
	exit(1);
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

int ParameterSet::getStringInt() const {
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
	return get<0>(intParams.at(n));
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

};	//End of namespace Util
};	//End of namespace TBTK
