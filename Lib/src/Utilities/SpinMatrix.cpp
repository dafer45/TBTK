/* Copyright 2017 Kristofer Björnson
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

/** @file SpinMatrix.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/SpinMatrix.h"

using namespace std;

namespace TBTK{

SpinMatrix::SpinMatrix(){
}

SpinMatrix::SpinMatrix(complex<double> value){
	for(unsigned int row = 0; row < getNumRows(); row++){
		for(unsigned int col = 0; col < getNumCols(); col++){
			at(row, col) = value;
		}
	}
}

SpinMatrix::~SpinMatrix(){
}

SpinMatrix& SpinMatrix::operator=(complex<double> value){
	for(unsigned int row = 0; row < getNumRows(); row++){
		for(unsigned int col = 0; col < getNumCols(); col++){
			at(row, col) = value;
		}
	}

	return *this;
}

SpinMatrix& SpinMatrix::operator+=(const SpinMatrix &spinMatrix){
	for(unsigned int row = 0; row < getNumRows(); row++){
		for(unsigned int col = 0; col < getNumCols(); col++){
			at(row, col) += spinMatrix.at(row, col);
		}
	}

	return *this;
}

SpinMatrix& SpinMatrix::operator-=(const SpinMatrix &spinMatrix){
	for(unsigned int row = 0; row < getNumRows(); row++){
		for(unsigned int col = 0; col < getNumCols(); col++){
			at(row, col) -= spinMatrix.at(row, col);
		}
	}

	return *this;
}

double SpinMatrix::getDensity() const{
	return abs(at(0, 0) + at(1, 1));
}

Vector3d SpinMatrix::getSpinVector() const{
	return Vector3d({
		real(at(0, 1) + at(1, 0)),
		imag(at(0, 1) - at(1, 0)),
		real(at(0, 0) - at(1, 1))
	});
}

std::string SpinMatrix::toString() const{
	stringstream stream;
	stream << "SpinMatrix\n";
	stream << "\t" << at(0, 0) << "\t" << at(0, 1) << "\n";
	stream << "\t" << at(1, 0) << "\t" << at(1, 1);

	return stream.str();
}

};	//End of namespace TBTK
