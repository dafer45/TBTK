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

/** @file SlaterKosterState.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/SlaterKosterState.h"
#include "TBTK/TBTKMacros.h"

#include <algorithm>

using namespace std;

namespace TBTK{

SlaterKosterState::SlaterKosterState(
) :
	AbstractState(AbstractState::StateID::SlaterKoster),
	radialFunction(nullptr)
{
}

SlaterKosterState::SlaterKosterState(
	const Vector3d &position,
	const string &orbital,
	const RadialFunction &radialFunction
) :
	AbstractState(AbstractState::StateID::SlaterKoster),
	position(position),
	orbital(getOrbital(orbital)),
	radialFunction(radialFunction.clone())
{
}

SlaterKosterState::SlaterKosterState(
	const Vector3d &position,
	Orbital orbital,
	const RadialFunction &radialFunction
) :
	AbstractState(AbstractState::StateID::SlaterKoster),
	position(position),
	orbital(orbital),
	radialFunction(radialFunction.clone())
{
}

SlaterKosterState::SlaterKosterState(
	const SlaterKosterState &slaterKosterState
) :
	AbstractState(AbstractState::StateID::SlaterKoster),
	position(slaterKosterState.position),
	orbital(slaterKosterState.orbital)
{
	if(slaterKosterState.radialFunction == nullptr)
		radialFunction = nullptr;
	else
		radialFunction = slaterKosterState.radialFunction->clone();
}

SlaterKosterState::~SlaterKosterState(){
	if(radialFunction != nullptr)
		delete radialFunction;
}

SlaterKosterState& SlaterKosterState::operator=(const SlaterKosterState &rhs){
	if(this != &rhs){
		position = rhs.position;
		orbital = rhs.orbital;

		if(radialFunction != nullptr)
			delete radialFunction;

		if(rhs.radialFunction == nullptr)
			radialFunction = nullptr;
		else
			radialFunction = rhs.radialFunction->clone();
	}

	return *this;
}

SlaterKosterState* SlaterKosterState::clone() const{
	return new SlaterKosterState(position, orbital, *radialFunction);
}

complex<double> SlaterKosterState::getOverlap(const AbstractState &bra) const{
	TBTKNotYetImplemented("SlaterKosterState::getOverlap()");
}

complex<double> SlaterKosterState::getMatrixElement(
	const AbstractState &bra,
	const AbstractOperator &o
) const{
	TBTKAssert(
		bra.getStateID() == AbstractState::SlaterKoster,
		"SlaterKosterState::getMatrixElement()",
		"Incompatible states.",
		"The bra state has to be a SlaterKosterState."
	);
	const SlaterKosterState &b = (const SlaterKosterState&)bra;
	Vector3d difference = position - b.position;
	double distance = difference.norm();
	double l = difference.x/difference.norm();
	double m = difference.y/difference.norm();
	double n = difference.z/difference.norm();

	switch(b.orbital){
	case Orbital::s:
		switch(orbital){
		case Orbital::s:
			return (*radialFunction)(
				distance,
				RadialFunction::Orbital::s,
				RadialFunction::Orbital::s,
				RadialFunction::Bond::Sigma
			);
		case Orbital::x:
			return l*(*radialFunction)(
				distance,
				RadialFunction::Orbital::s,
				RadialFunction::Orbital::p,
				RadialFunction::Bond::Sigma
			);
		case Orbital::y:
			return m*(*radialFunction)(
				distance,
				RadialFunction::Orbital::s,
				RadialFunction::Orbital::p,
				RadialFunction::Bond::Sigma
			);
		case Orbital::z:
			return n*(*radialFunction)(
				distance,
				RadialFunction::Orbital::s,
				RadialFunction::Orbital::p,
				RadialFunction::Bond::Sigma
			);
		case Orbital::xy:
			return sqrt(3.)*l*m*(*radialFunction)(
				distance,
				RadialFunction::Orbital::s,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			);
		case Orbital::yz:
			return sqrt(3.)*m*n*(*radialFunction)(
				distance,
				RadialFunction::Orbital::s,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			);
		case Orbital::zx:
			return sqrt(3.)*n*l*(*radialFunction)(
				distance,
				RadialFunction::Orbital::s,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			);
		case Orbital::x2my2:
			return (sqrt(3.)/2.)*(l*l - m*m)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::s,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			);
		case Orbital::z2mr2:
			return (n*n - (l*l + m*m)/2.)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::s,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			);
		default:
			TBTKExit(
				"SlaterKosterState::getMatrixElement()",
				"Unknown orbital. This should never happen,"
				<< " contact the developer.",
				""
			);
		}
	case Orbital::x:
		switch(orbital){
		case Orbital::s:
			return -l*(*radialFunction)(
				distance,
				RadialFunction::Orbital::s,
				RadialFunction::Orbital::p,
				RadialFunction::Bond::Sigma
			);
		case Orbital::x:
			return l*l*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::p,
				RadialFunction::Bond::Sigma
			) + (1 - l*l)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::p,
				RadialFunction::Bond::Pi
			);
		case Orbital::y:
			return l*m*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::p,
				RadialFunction::Bond::Sigma
			) - l*m*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::p,
				RadialFunction::Bond::Pi
			);
		case Orbital::z:
			return l*n*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::p,
				RadialFunction::Bond::Sigma
			) - l*n*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::p,
				RadialFunction::Bond::Pi
			);
		case Orbital::xy:
			return sqrt(3.)*l*l*m*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			) + m*(1 - 2*l*l)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Pi
			);
		case Orbital::yz:
			return sqrt(3.)*l*m*n*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			) - 2*l*m*n*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Pi
			);
		case Orbital::zx:
			return sqrt(3.)*l*l*n*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			) + n*(1 - 2*l*l)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Pi
			);
		case Orbital::x2my2:
			return (sqrt(3.)/2.)*l*(l*l - m*m)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			) + l*(1 - l*l + m*m)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Pi
			);
		case Orbital::z2mr2:
			return l*(n*n - (l*l + m*m)/2.)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			) - sqrt(3.)*l*n*n*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Pi
			);
		default:
			TBTKExit(
				"SlaterKosterState::getMatrixElement()",
				"Unknown orbital. This should never happen,"
				<< " contact the developer.",
				""
			);
		}
	case Orbital::y:
		switch(orbital){
		case Orbital::s:
			return -m*(*radialFunction)(
				distance,
				RadialFunction::Orbital::s,
				RadialFunction::Orbital::p,
				RadialFunction::Bond::Sigma
			);
		case Orbital::x:
			return m*l*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::p,
				RadialFunction::Bond::Sigma
			) - m*l*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::p,
				RadialFunction::Bond::Pi
			);
		case Orbital::y:
			return m*m*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::p,
				RadialFunction::Bond::Sigma
			) + (1 - m*m)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::p,
				RadialFunction::Bond::Pi
			);
		case Orbital::z:
			return m*n*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::p,
				RadialFunction::Bond::Sigma
			) - m*n*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::p,
				RadialFunction::Bond::Pi
			);
		case Orbital::xy:
			return sqrt(3.)*m*m*l*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			) + l*(1 - 2*m*m)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Pi
			);
		case Orbital::yz:
			return sqrt(3.)*m*m*n*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			) + n*(1 - 2*m*m)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Pi
			);
		case Orbital::zx:
			return sqrt(3.)*m*n*l*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			) - 2*m*n*l*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Pi
			);
		case Orbital::x2my2:
			return (sqrt(3.)/2.)*m*(l*l - m*m)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			) - m*(1 + l*l - m*m)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Pi
			);
		case Orbital::z2mr2:
			return m*(n*n - (l*l + m*m)/2.)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			) - sqrt(3.)*m*n*n*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Pi
			);
		default:
			TBTKExit(
				"SlaterKosterState::getMatrixElement()",
				"Unknown orbital. This should never happen,"
				<< " contact the developer.",
				""
			);
		}
	case Orbital::z:
		switch(orbital){
		case Orbital::s:
			return -n*(*radialFunction)(
				distance,
				RadialFunction::Orbital::s,
				RadialFunction::Orbital::p,
				RadialFunction::Bond::Sigma
			);
		case Orbital::x:
			return n*l*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::p,
				RadialFunction::Bond::Sigma
			) - n*l*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::p,
				RadialFunction::Bond::Pi
			);
		case Orbital::y:
			return n*m*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::p,
				RadialFunction::Bond::Sigma
			) - n*m*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::p,
				RadialFunction::Bond::Pi
			);
		case Orbital::z:
			return n*n*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::p,
				RadialFunction::Bond::Sigma
			) + (1 - n*n)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::p,
				RadialFunction::Bond::Pi
			);
		case Orbital::xy:
			return sqrt(3.)*n*l*m*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			) - 2*n*l*m*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Pi
			);
		case Orbital::yz:
			return sqrt(3.)*n*n*m*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			) + m*(1 - 2*n*n)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Pi
			);
		case Orbital::zx:
			return sqrt(3.)*n*n*l*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			) + l*(1 - 2*n*n)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Pi
			);
		case Orbital::x2my2:
			return (sqrt(3.)/2.)*n*(l*l - m*m)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			) - n*(l*l - m*m)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Pi
			);
		case Orbital::z2mr2:
			return n*(n*n - (l*l + m*m)/2.)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			) + sqrt(3.)*n*(l*l + m*m)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Pi
			);
		default:
			TBTKExit(
				"SlaterKosterState::getMatrixElement()",
				"Unknown orbital. This should never happen,"
				<< " contact the developer.",
				""
			);
		}
	case Orbital::xy:
		switch(orbital){
		case Orbital::s:
			return sqrt(3.)*l*m*(*radialFunction)(
				distance,
				RadialFunction::Orbital::s,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			);
		case Orbital::x:
			return -sqrt(3.)*l*l*m*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			) - m*(1 - 2*l*l)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Pi
			);
		case Orbital::y:
			return -sqrt(3.)*m*m*l*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			) - l*(1 - 2*m*m)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Pi
			);
		case Orbital::z:
			return -sqrt(3.)*n*l*m*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			) + 2*n*l*m*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Pi
			);
		case Orbital::xy:
			return 3*l*l*m*m*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			) + (l*l + m*m - 4*l*l*m*m)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Pi
			) + (n*n + l*l*m*m)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Delta
			);
		case Orbital::yz:
			return 3*l*m*m*n*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			) + l*n*(1 - 4*m*m)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Pi
			) + l*n*(m*m - 1)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Delta
			);
		case Orbital::zx:
			return 3*l*l*m*n*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			) + m*n*(1 - 4*l*l)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Pi
			) + m*n*(l*l - 1)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Delta
			);
		case Orbital::x2my2:
			return (3./2.)*l*m*(l*l - m*m)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			) + 2*l*m*(m*m - l*l)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Pi
			) + (l*m*(l*l - m*m)/2.)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Delta
			);
		case Orbital::z2mr2:
			return sqrt(3.)*(
				l*m*(n*n - (l*l + m*m)/2.)*(*radialFunction)(
					distance,
					RadialFunction::Orbital::d,
					RadialFunction::Orbital::d,
					RadialFunction::Bond::Sigma
				) - 2*l*m*n*n*(*radialFunction)(
					distance,
					RadialFunction::Orbital::d,
					RadialFunction::Orbital::d,
					RadialFunction::Bond::Pi
				) + (l*m*(1 + n*n)/2.)*(*radialFunction)(
					distance,
					RadialFunction::Orbital::d,
					RadialFunction::Orbital::d,
					RadialFunction::Bond::Delta
				)
			);
		default:
			TBTKExit(
				"SlaterKosterState::getMatrixElement()",
				"Unknown orbital. This should never happen,"
				<< " contact the developer.",
				""
			);
		}
	case Orbital::yz:
		switch(orbital){
		case Orbital::s:
			return sqrt(3.)*m*n*(*radialFunction)(
				distance,
				RadialFunction::Orbital::s,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			);
		case Orbital::x:
			return -sqrt(3.)*l*m*n*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			) + 2*l*m*n*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Pi
			);
		case Orbital::y:
			return -sqrt(3.)*m*m*n*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			) - n*(1 - 2*m*m)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Pi
			);
		case Orbital::z:
			return -sqrt(3.)*n*n*m*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			) - m*(1 - 2*n*n)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Pi
			);
		case Orbital::xy:
			return 3*l*m*m*n*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			) + l*n*(1 - 4*m*m)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Pi
			) + l*n*(m*m - 1)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Delta
			);
		case Orbital::yz:
			return 3*m*m*n*n*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			) + (m*m + n*n - 4*m*m*n*n)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Pi
			) + (l*l + m*m*n*n)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Delta
			);
		case Orbital::zx:
			return 3*m*n*n*l*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			) + m*l*(1 - 4*n*n)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Pi
			) + m*l*(n*n - 1)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Delta
			);
		case Orbital::x2my2:
			return (3./2.)*m*n*(l*l - m*m)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			) - m*n*(1 + 2*(l*l - m*m))*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Pi
			) + m*n*(1 + (l*l - m*m)/2.)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Delta
			);
		case Orbital::z2mr2:
			return sqrt(3.)*(
				m*n*(n*n - (l*l + m*m)/2.)*(*radialFunction)(
					distance,
					RadialFunction::Orbital::d,
					RadialFunction::Orbital::d,
					RadialFunction::Bond::Sigma
				) + m*n*(l*l + m*m - n*n)*(*radialFunction)(
					distance,
					RadialFunction::Orbital::d,
					RadialFunction::Orbital::d,
					RadialFunction::Bond::Pi
				) - (m*n*(l*l + m*m)/2.)*(*radialFunction)(
					distance,
					RadialFunction::Orbital::d,
					RadialFunction::Orbital::d,
					RadialFunction::Bond::Delta
				)
			);
		default:
			TBTKExit(
				"SlaterKosterState::getMatrixElement()",
				"Unknown orbital. This should never happen,"
				<< " contact the developer.",
				""
			);
		}
	case Orbital::zx:
		switch(orbital){
		case Orbital::s:
			return sqrt(3.)*n*l*(*radialFunction)(
				distance,
				RadialFunction::Orbital::s,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			);
		case Orbital::x:
			return -sqrt(3.)*l*l*n*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			) - n*(1 - 2*l*l)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Pi
			);
		case Orbital::y:
			return -sqrt(3.)*m*n*l*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			) + 2*m*n*l*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Pi
			);
		case Orbital::z:
			return -sqrt(3.)*n*n*l*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			) - l*(1 - 2*n*n)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Pi
			);
		case Orbital::xy:
			return 3*l*l*m*n*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			) + m*n*(1 - 4*l*l)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Pi
			) + m*n*(l*l - 1)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Delta
			);
		case Orbital::yz:
			return 3*m*n*n*l*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			) + m*l*(1 - 4*n*n)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Pi
			) + m*l*(n*n - 1)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Delta
			);
		case Orbital::zx:
			return 3*n*n*l*l*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			) + (n*n + l*l - 4*n*n*l*l)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Pi
			) + (m*m + n*n*l*l)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Delta
			);
		case Orbital::x2my2:
			return (3./2.)*n*l*(l*l - m*m)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			) + n*l*(1 - 2*(l*l - m*m))*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Pi
			) - n*l*(1 - (l*l - m*m)/2.)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Delta
			);
		case Orbital::z2mr2:
			return sqrt(3.)*(
				l*n*(n*n - (l*l + m*m)/2.)*(*radialFunction)(
					distance,
					RadialFunction::Orbital::d,
					RadialFunction::Orbital::d,
					RadialFunction::Bond::Sigma
				) + l*n*(l*l + m*m - n*n)*(*radialFunction)(
					distance,
					RadialFunction::Orbital::d,
					RadialFunction::Orbital::d,
					RadialFunction::Bond::Pi
				) - (l*n*(l*l + m*m)/2.)*(*radialFunction)(
					distance,
					RadialFunction::Orbital::d,
					RadialFunction::Orbital::d,
					RadialFunction::Bond::Delta
				)
			);
		default:
			TBTKExit(
				"SlaterKosterState::getMatrixElement()",
				"Unknown orbital. This should never happen,"
				<< " contact the developer.",
				""
			);
		}
	case Orbital::x2my2:
		switch(orbital){
		case Orbital::s:
			return (sqrt(3.)/2.)*(l*l - m*m)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::s,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			);
		case Orbital::x:
			return -(sqrt(3.)/2.)*l*(l*l - m*m)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			) - l*(1 - l*l + m*m)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Pi
			);
		case Orbital::y:
			return -(sqrt(3.)/2.)*m*(l*l - m*m)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			) + m*(1 + l*l - m*m)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Pi
			);
		case Orbital::z:
			return -(sqrt(3.)/2.)*n*(l*l - m*m)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			) + n*(l*l - m*m)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Pi
			);
		case Orbital::xy:
			return (3./2.)*l*m*(l*l - m*m)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			) + 2*l*m*(m*m - l*l)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Pi
			) + (l*m*(l*l - m*m)/2.)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Delta
			);
		case Orbital::yz:
			return (3./2.)*m*n*(l*l - m*m)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			) - m*n*(1 + 2*(l*l - m*m))*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Pi
			) + m*n*(1 + (l*l - m*m)/2.)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Delta
			);
		case Orbital::zx:
			return (3./2.)*n*l*(l*l - m*m)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			) + n*l*(1 - 2*(l*l - m*m))*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Pi
			) - n*l*(1 - (l*l - m*m)/2.)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Delta
			);
		case Orbital::x2my2:
			return (3./4.)*(l*l - m*m)*(l*l - m*m)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			) + (l*l + m*m - (l*l - m*m)*(l*l - m*m))*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Pi
			) + (n*n + (l*l - m*m)*(l*l - m*m)/4.)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Delta
			);
		case Orbital::z2mr2:
			return sqrt(3.)*(
				(l*l - m*m)*(n*n - (l*l + m*m)/2.)*(*radialFunction)(
					distance,
					RadialFunction::Orbital::d,
					RadialFunction::Orbital::d,
					RadialFunction::Bond::Sigma
				)/2. + n*n*(m*m - l*l)*(*radialFunction)(
					distance,
					RadialFunction::Orbital::d,
					RadialFunction::Orbital::d,
					RadialFunction::Bond::Pi
				) + ((1 + n*n)*(l*l - m*m)/4.)*(*radialFunction)(
					distance,
					RadialFunction::Orbital::d,
					RadialFunction::Orbital::d,
					RadialFunction::Bond::Delta
				)
			);
		default:
			TBTKExit(
				"SlaterKosterState::getMatrixElement()",
				"Unknown orbital. This should never happen,"
				<< " contact the developer.",
				""
			);
		}
	case Orbital::z2mr2:
		switch(orbital){
		case Orbital::s:
			return (n*n - (l*l + m*m)/2.)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::s,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			);
		case Orbital::x:
			return -l*(n*n - (l*l + m*m)/2.)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			) + sqrt(3.)*l*n*n*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Pi
			);
		case Orbital::y:
			return -m*(n*n - (l*l + m*m)/2.)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			) + sqrt(3.)*m*n*n*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Pi
			);
		case Orbital::z:
			return -n*(n*n - (l*l + m*m)/2.)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			) - sqrt(3.)*n*(l*l + m*m)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::p,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Pi
			);
		case Orbital::xy:
			return sqrt(3.)*(
				l*m*(n*n - (l*l + m*m)/2.)*(*radialFunction)(
					distance,
					RadialFunction::Orbital::d,
					RadialFunction::Orbital::d,
					RadialFunction::Bond::Sigma
				) - 2*l*m*n*n*(*radialFunction)(
					distance,
					RadialFunction::Orbital::d,
					RadialFunction::Orbital::d,
					RadialFunction::Bond::Pi
				) + (l*m*(1 + n*n)/2.)*(*radialFunction)(
					distance,
					RadialFunction::Orbital::d,
					RadialFunction::Orbital::d,
					RadialFunction::Bond::Delta
				)
			);
		case Orbital::yz:
			return sqrt(3.)*(
				m*n*(n*n - (l*l + m*m)/2.)*(*radialFunction)(
					distance,
					RadialFunction::Orbital::d,
					RadialFunction::Orbital::d,
					RadialFunction::Bond::Sigma
				) + m*n*(l*l + m*m - n*n)*(*radialFunction)(
					distance,
					RadialFunction::Orbital::d,
					RadialFunction::Orbital::d,
					RadialFunction::Bond::Pi
				) - (m*n*(l*l + m*m)/2.)*(*radialFunction)(
					distance,
					RadialFunction::Orbital::d,
					RadialFunction::Orbital::d,
					RadialFunction::Bond::Delta
				)
			);
		case Orbital::zx:
			return sqrt(3.)*(
				l*n*(n*n - (l*l + m*m)/2.)*(*radialFunction)(
					distance,
					RadialFunction::Orbital::d,
					RadialFunction::Orbital::d,
					RadialFunction::Bond::Sigma
				) + l*n*(l*l + m*m - n*n)*(*radialFunction)(
					distance,
					RadialFunction::Orbital::d,
					RadialFunction::Orbital::d,
					RadialFunction::Bond::Pi
				) - (l*n*(l*l + m*m)/2.)*(*radialFunction)(
					distance,
					RadialFunction::Orbital::d,
					RadialFunction::Orbital::d,
					RadialFunction::Bond::Delta
				)
			);
		case Orbital::x2my2:
			return sqrt(3.)*(
				(l*l - m*m)*(n*n - (l*l + m*m)/2.)*(*radialFunction)(
					distance,
					RadialFunction::Orbital::d,
					RadialFunction::Orbital::d,
					RadialFunction::Bond::Sigma
				)/2. + n*n*(m*m - l*l)*(*radialFunction)(
					distance,
					RadialFunction::Orbital::d,
					RadialFunction::Orbital::d,
					RadialFunction::Bond::Pi
				) + ((1 + n*n)*(l*l - m*m)/4.)*(*radialFunction)(
					distance,
					RadialFunction::Orbital::d,
					RadialFunction::Orbital::d,
					RadialFunction::Bond::Delta
				)
			);
		case Orbital::z2mr2:
			return pow(n*n - (l*l + m*m)/2., 2)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Sigma
			) + 3*n*n*(l*l + m*m)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Pi
			) + (3./4.)*pow(l*l +m*m, 2)*(*radialFunction)(
				distance,
				RadialFunction::Orbital::d,
				RadialFunction::Orbital::d,
				RadialFunction::Bond::Delta
			);
		default:
			TBTKExit(
				"SlaterKosterState::getMatrixElement()",
				"Unknown orbital. This should never happen,"
				<< " contact the developer.",
				""
			);
		}
	default:
		TBTKExit(
			"SlaterKosterState::getMatrixElement()",
			"Unknown orbital. This should never happen, contact"
			<< " the developer.",
			""
		);
	}
}

SlaterKosterState::Orbital SlaterKosterState::getOrbital(
	const string &orbital
){
	if(orbital.compare("s") == 0)
		return Orbital::s;
	else if(orbital.compare("x") == 0)
		return Orbital::x;
	else if(orbital.compare("y") == 0)
		return Orbital::y;
	else if(orbital.compare("z") == 0)
		return Orbital::z;
	else if(orbital.compare("xy") == 0)
		return Orbital::xy;
	else if(orbital.compare("yz") == 0)
		return Orbital::yz;
	else if(orbital.compare("zx") == 0)
		return Orbital::zx;
	else if(orbital.compare("x^2-y^2") == 0)
		return Orbital::x2my2;
	else if(orbital.compare("3z^2-r^2") == 0)
		return Orbital::z2mr2;

	TBTKExit(
		"SlaterKosterState::getOrbital()",
		"Unknown orbital '" << orbital << "'.",
		""
	);
}

SlaterKosterState::RadialFunction::~RadialFunction(){
}

};	//End of namespace TBTK
