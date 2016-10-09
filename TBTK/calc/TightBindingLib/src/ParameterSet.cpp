/** @file ParameterSet.cpp
 *
 *  @author Kristofer Björnson
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

int ParameterSet::getInt(string name){
	for(unsigned int n = 0; n < intParams.size(); n++){
		if((get<0>(intParams.at(n))).compare(name) == 0){
			return get<1>(intParams.at(n));
		}
	}

	cout << "Error in ParameterSet::getInt(): Parameter '" << name << "' not defined.\n";
	exit(1);
}

double ParameterSet::getDouble(string name){
	for(unsigned int n = 0; n < doubleParams.size(); n++){
		if((get<0>(doubleParams.at(n))).compare(name) == 0){
			return get<1>(doubleParams.at(n));
		}
	}

	cout << "Error in ParameterSet::getDouble(): Parameter '" << name << "' not defined.\n";
	exit(1);
}

complex<double> ParameterSet::getComplex(string name){
	for(unsigned int n = 0; n < complexParams.size(); n++){
		if((get<0>(complexParams.at(n))).compare(name) == 0){
			return get<1>(complexParams.at(n));
		}
	}

	cout << "Error in ParameterSet::getComplex(): Parameter '" << name << "' not defined.\n";
	exit(1);
}

int ParameterSet::getNumInt(){
	return intParams.size();
}

int ParameterSet::getNumDouble(){
	return doubleParams.size();
}

int ParameterSet::getNumComplex(){
	return complexParams.size();
}

std::string ParameterSet::getIntName(int n){
	return get<0>(intParams.at(n));
}

std::string ParameterSet::getDoubleName(int n){
	return get<0>(doubleParams.at(n));
}

std::string ParameterSet::getComplexName(int n){
	return get<0>(complexParams.at(n));
}

int ParameterSet::getIntValue(int n){
	return get<1>(intParams.at(n));
}

double ParameterSet::getDoubleValue(int n){
	return get<1>(doubleParams.at(n));
}

complex<double> ParameterSet::getComplexValue(int n){
	return get<1>(complexParams.at(n));
}

};	//End of namespace Util
};	//End of namespace TBTK
