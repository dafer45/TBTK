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
	parametrization(nullptr)
{
}

SlaterKosterState::SlaterKosterState(
	const Vector3d &position,
	const string &orbital,
	const Parametrization &parametrization
) :
	AbstractState(AbstractState::StateID::SlaterKoster),
	position(position),
	orbital(getOrbital(orbital)),
	parametrization(parametrization.clone())
{
}

SlaterKosterState::SlaterKosterState(
	const Vector3d &position,
	Orbital orbital,
	const Parametrization &parametrization
) :
	AbstractState(AbstractState::StateID::SlaterKoster),
	position(position),
	orbital(orbital),
	parametrization(parametrization.clone())
{
}

SlaterKosterState::SlaterKosterState(
	const SlaterKosterState &slaterKosterState
) :
	AbstractState(AbstractState::StateID::SlaterKoster),
	position(slaterKosterState.position),
	orbital(slaterKosterState.orbital)
{
	if(slaterKosterState.parametrization == nullptr)
		parametrization = nullptr;
	else
		parametrization = slaterKosterState.parametrization->clone();
}

SlaterKosterState::~SlaterKosterState(){
	if(parametrization != nullptr)
		delete parametrization;
}

SlaterKosterState& SlaterKosterState::operator=(const SlaterKosterState &rhs){
	if(this != &rhs){
		position = rhs.position;
		orbital = rhs.orbital;

		if(parametrization != nullptr)
			delete parametrization;

		if(rhs.parametrization == nullptr)
			parametrization = nullptr;
		else
			parametrization = rhs.parametrization->clone();
	}

	return *this;
}

SlaterKosterState* SlaterKosterState::clone() const{
	return new SlaterKosterState(position, orbital, *parametrization);
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
	if(distance < numeric_limits<double>::epsilon()){
		if(b.orbital != orbital)
			return 0;
		switch(orbital){
		case Orbital::s:
			return parametrization->getOnSiteTerm(
				Parametrization::Orbital::s
			);
		case Orbital::x:
		case Orbital::y:
		case Orbital::z:
			return parametrization->getOnSiteTerm(
				Parametrization::Orbital::p
			);
		case Orbital::xy:
		case Orbital::yz:
		case Orbital::zx:
		case Orbital::x2my2:
		case Orbital::z2mr2:
			return parametrization->getOnSiteTerm(
				Parametrization::Orbital::d
			);
		}
	}
	double l = difference.x/difference.norm();
	double m = difference.y/difference.norm();
	double n = difference.z/difference.norm();

	switch(b.orbital){
	case Orbital::s:
		switch(orbital){
		case Orbital::s:
			return parametrization->getParameter(
				distance,
				Parametrization::Orbital::s,
				Parametrization::Orbital::s,
				Parametrization::Bond::Sigma
			);
		case Orbital::x:
			return l*parametrization->getParameter(
				distance,
				Parametrization::Orbital::s,
				Parametrization::Orbital::p,
				Parametrization::Bond::Sigma
			);
		case Orbital::y:
			return m*parametrization->getParameter(
				distance,
				Parametrization::Orbital::s,
				Parametrization::Orbital::p,
				Parametrization::Bond::Sigma
			);
		case Orbital::z:
			return n*parametrization->getParameter(
				distance,
				Parametrization::Orbital::s,
				Parametrization::Orbital::p,
				Parametrization::Bond::Sigma
			);
		case Orbital::xy:
			return sqrt(3.)*l*m*parametrization->getParameter(
				distance,
				Parametrization::Orbital::s,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			);
		case Orbital::yz:
			return sqrt(3.)*m*n*parametrization->getParameter(
				distance,
				Parametrization::Orbital::s,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			);
		case Orbital::zx:
			return sqrt(3.)*n*l*parametrization->getParameter(
				distance,
				Parametrization::Orbital::s,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			);
		case Orbital::x2my2:
			return (sqrt(3.)/2.)*(l*l - m*m)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::s,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			);
		case Orbital::z2mr2:
			return (n*n - (l*l + m*m)/2.)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::s,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
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
			return -l*parametrization->getParameter(
				distance,
				Parametrization::Orbital::s,
				Parametrization::Orbital::p,
				Parametrization::Bond::Sigma
			);
		case Orbital::x:
			return l*l*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::p,
				Parametrization::Bond::Sigma
			) + (1 - l*l)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::p,
				Parametrization::Bond::Pi
			);
		case Orbital::y:
			return l*m*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::p,
				Parametrization::Bond::Sigma
			) - l*m*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::p,
				Parametrization::Bond::Pi
			);
		case Orbital::z:
			return l*n*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::p,
				Parametrization::Bond::Sigma
			) - l*n*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::p,
				Parametrization::Bond::Pi
			);
		case Orbital::xy:
			return sqrt(3.)*l*l*m*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			) + m*(1 - 2*l*l)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Pi
			);
		case Orbital::yz:
			return sqrt(3.)*l*m*n*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			) - 2*l*m*n*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Pi
			);
		case Orbital::zx:
			return sqrt(3.)*l*l*n*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			) + n*(1 - 2*l*l)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Pi
			);
		case Orbital::x2my2:
			return (sqrt(3.)/2.)*l*(l*l - m*m)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			) + l*(1 - l*l + m*m)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Pi
			);
		case Orbital::z2mr2:
			return l*(n*n - (l*l + m*m)/2.)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			) - sqrt(3.)*l*n*n*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Pi
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
			return -m*parametrization->getParameter(
				distance,
				Parametrization::Orbital::s,
				Parametrization::Orbital::p,
				Parametrization::Bond::Sigma
			);
		case Orbital::x:
			return m*l*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::p,
				Parametrization::Bond::Sigma
			) - m*l*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::p,
				Parametrization::Bond::Pi
			);
		case Orbital::y:
			return m*m*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::p,
				Parametrization::Bond::Sigma
			) + (1 - m*m)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::p,
				Parametrization::Bond::Pi
			);
		case Orbital::z:
			return m*n*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::p,
				Parametrization::Bond::Sigma
			) - m*n*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::p,
				Parametrization::Bond::Pi
			);
		case Orbital::xy:
			return sqrt(3.)*m*m*l*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			) + l*(1 - 2*m*m)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Pi
			);
		case Orbital::yz:
			return sqrt(3.)*m*m*n*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			) + n*(1 - 2*m*m)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Pi
			);
		case Orbital::zx:
			return sqrt(3.)*m*n*l*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			) - 2*m*n*l*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Pi
			);
		case Orbital::x2my2:
			return (sqrt(3.)/2.)*m*(l*l - m*m)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			) - m*(1 + l*l - m*m)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Pi
			);
		case Orbital::z2mr2:
			return m*(n*n - (l*l + m*m)/2.)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			) - sqrt(3.)*m*n*n*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Pi
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
			return -n*parametrization->getParameter(
				distance,
				Parametrization::Orbital::s,
				Parametrization::Orbital::p,
				Parametrization::Bond::Sigma
			);
		case Orbital::x:
			return n*l*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::p,
				Parametrization::Bond::Sigma
			) - n*l*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::p,
				Parametrization::Bond::Pi
			);
		case Orbital::y:
			return n*m*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::p,
				Parametrization::Bond::Sigma
			) - n*m*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::p,
				Parametrization::Bond::Pi
			);
		case Orbital::z:
			return n*n*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::p,
				Parametrization::Bond::Sigma
			) + (1 - n*n)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::p,
				Parametrization::Bond::Pi
			);
		case Orbital::xy:
			return sqrt(3.)*n*l*m*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			) - 2*n*l*m*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Pi
			);
		case Orbital::yz:
			return sqrt(3.)*n*n*m*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			) + m*(1 - 2*n*n)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Pi
			);
		case Orbital::zx:
			return sqrt(3.)*n*n*l*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			) + l*(1 - 2*n*n)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Pi
			);
		case Orbital::x2my2:
			return (sqrt(3.)/2.)*n*(l*l - m*m)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			) - n*(l*l - m*m)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Pi
			);
		case Orbital::z2mr2:
			return n*(n*n - (l*l + m*m)/2.)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			) + sqrt(3.)*n*(l*l + m*m)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Pi
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
			return sqrt(3.)*l*m*parametrization->getParameter(
				distance,
				Parametrization::Orbital::s,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			);
		case Orbital::x:
			return -sqrt(3.)*l*l*m*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			) - m*(1 - 2*l*l)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Pi
			);
		case Orbital::y:
			return -sqrt(3.)*m*m*l*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			) - l*(1 - 2*m*m)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Pi
			);
		case Orbital::z:
			return -sqrt(3.)*n*l*m*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			) + 2*n*l*m*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Pi
			);
		case Orbital::xy:
			return 3*l*l*m*m*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			) + (l*l + m*m - 4*l*l*m*m)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Pi
			) + (n*n + l*l*m*m)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Delta
			);
		case Orbital::yz:
			return 3*l*m*m*n*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			) + l*n*(1 - 4*m*m)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Pi
			) + l*n*(m*m - 1)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Delta
			);
		case Orbital::zx:
			return 3*l*l*m*n*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			) + m*n*(1 - 4*l*l)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Pi
			) + m*n*(l*l - 1)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Delta
			);
		case Orbital::x2my2:
			return (3./2.)*l*m*(l*l - m*m)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			) + 2*l*m*(m*m - l*l)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Pi
			) + (l*m*(l*l - m*m)/2.)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Delta
			);
		case Orbital::z2mr2:
			return sqrt(3.)*(
				l*m*(n*n - (l*l + m*m)/2.)*parametrization->getParameter(
					distance,
					Parametrization::Orbital::d,
					Parametrization::Orbital::d,
					Parametrization::Bond::Sigma
				) - 2*l*m*n*n*parametrization->getParameter(
					distance,
					Parametrization::Orbital::d,
					Parametrization::Orbital::d,
					Parametrization::Bond::Pi
				) + (l*m*(1 + n*n)/2.)*parametrization->getParameter(
					distance,
					Parametrization::Orbital::d,
					Parametrization::Orbital::d,
					Parametrization::Bond::Delta
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
			return sqrt(3.)*m*n*parametrization->getParameter(
				distance,
				Parametrization::Orbital::s,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			);
		case Orbital::x:
			return -sqrt(3.)*l*m*n*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			) + 2*l*m*n*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Pi
			);
		case Orbital::y:
			return -sqrt(3.)*m*m*n*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			) - n*(1 - 2*m*m)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Pi
			);
		case Orbital::z:
			return -sqrt(3.)*n*n*m*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			) - m*(1 - 2*n*n)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Pi
			);
		case Orbital::xy:
			return 3*l*m*m*n*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			) + l*n*(1 - 4*m*m)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Pi
			) + l*n*(m*m - 1)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Delta
			);
		case Orbital::yz:
			return 3*m*m*n*n*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			) + (m*m + n*n - 4*m*m*n*n)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Pi
			) + (l*l + m*m*n*n)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Delta
			);
		case Orbital::zx:
			return 3*m*n*n*l*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			) + m*l*(1 - 4*n*n)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Pi
			) + m*l*(n*n - 1)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Delta
			);
		case Orbital::x2my2:
			return (3./2.)*m*n*(l*l - m*m)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			) - m*n*(1 + 2*(l*l - m*m))*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Pi
			) + m*n*(1 + (l*l - m*m)/2.)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Delta
			);
		case Orbital::z2mr2:
			return sqrt(3.)*(
				m*n*(n*n - (l*l + m*m)/2.)*parametrization->getParameter(
					distance,
					Parametrization::Orbital::d,
					Parametrization::Orbital::d,
					Parametrization::Bond::Sigma
				) + m*n*(l*l + m*m - n*n)*parametrization->getParameter(
					distance,
					Parametrization::Orbital::d,
					Parametrization::Orbital::d,
					Parametrization::Bond::Pi
				) - (m*n*(l*l + m*m)/2.)*parametrization->getParameter(
					distance,
					Parametrization::Orbital::d,
					Parametrization::Orbital::d,
					Parametrization::Bond::Delta
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
			return sqrt(3.)*n*l*parametrization->getParameter(
				distance,
				Parametrization::Orbital::s,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			);
		case Orbital::x:
			return -sqrt(3.)*l*l*n*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			) - n*(1 - 2*l*l)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Pi
			);
		case Orbital::y:
			return -sqrt(3.)*m*n*l*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			) + 2*m*n*l*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Pi
			);
		case Orbital::z:
			return -sqrt(3.)*n*n*l*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			) - l*(1 - 2*n*n)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Pi
			);
		case Orbital::xy:
			return 3*l*l*m*n*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			) + m*n*(1 - 4*l*l)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Pi
			) + m*n*(l*l - 1)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Delta
			);
		case Orbital::yz:
			return 3*m*n*n*l*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			) + m*l*(1 - 4*n*n)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Pi
			) + m*l*(n*n - 1)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Delta
			);
		case Orbital::zx:
			return 3*n*n*l*l*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			) + (n*n + l*l - 4*n*n*l*l)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Pi
			) + (m*m + n*n*l*l)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Delta
			);
		case Orbital::x2my2:
			return (3./2.)*n*l*(l*l - m*m)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			) + n*l*(1 - 2*(l*l - m*m))*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Pi
			) - n*l*(1 - (l*l - m*m)/2.)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Delta
			);
		case Orbital::z2mr2:
			return sqrt(3.)*(
				l*n*(n*n - (l*l + m*m)/2.)*parametrization->getParameter(
					distance,
					Parametrization::Orbital::d,
					Parametrization::Orbital::d,
					Parametrization::Bond::Sigma
				) + l*n*(l*l + m*m - n*n)*parametrization->getParameter(
					distance,
					Parametrization::Orbital::d,
					Parametrization::Orbital::d,
					Parametrization::Bond::Pi
				) - (l*n*(l*l + m*m)/2.)*parametrization->getParameter(
					distance,
					Parametrization::Orbital::d,
					Parametrization::Orbital::d,
					Parametrization::Bond::Delta
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
			return (sqrt(3.)/2.)*(l*l - m*m)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::s,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			);
		case Orbital::x:
			return -(sqrt(3.)/2.)*l*(l*l - m*m)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			) - l*(1 - l*l + m*m)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Pi
			);
		case Orbital::y:
			return -(sqrt(3.)/2.)*m*(l*l - m*m)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			) + m*(1 + l*l - m*m)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Pi
			);
		case Orbital::z:
			return -(sqrt(3.)/2.)*n*(l*l - m*m)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			) + n*(l*l - m*m)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Pi
			);
		case Orbital::xy:
			return (3./2.)*l*m*(l*l - m*m)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			) + 2*l*m*(m*m - l*l)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Pi
			) + (l*m*(l*l - m*m)/2.)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Delta
			);
		case Orbital::yz:
			return (3./2.)*m*n*(l*l - m*m)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			) - m*n*(1 + 2*(l*l - m*m))*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Pi
			) + m*n*(1 + (l*l - m*m)/2.)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Delta
			);
		case Orbital::zx:
			return (3./2.)*n*l*(l*l - m*m)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			) + n*l*(1 - 2*(l*l - m*m))*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Pi
			) - n*l*(1 - (l*l - m*m)/2.)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Delta
			);
		case Orbital::x2my2:
			return (3./4.)*(l*l - m*m)*(l*l - m*m)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			) + (l*l + m*m - (l*l - m*m)*(l*l - m*m))*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Pi
			) + (n*n + (l*l - m*m)*(l*l - m*m)/4.)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Delta
			);
		case Orbital::z2mr2:
			return sqrt(3.)*(
				(l*l - m*m)*(n*n - (l*l + m*m)/2.)*parametrization->getParameter(
					distance,
					Parametrization::Orbital::d,
					Parametrization::Orbital::d,
					Parametrization::Bond::Sigma
				)/2. + n*n*(m*m - l*l)*parametrization->getParameter(
					distance,
					Parametrization::Orbital::d,
					Parametrization::Orbital::d,
					Parametrization::Bond::Pi
				) + ((1 + n*n)*(l*l - m*m)/4.)*parametrization->getParameter(
					distance,
					Parametrization::Orbital::d,
					Parametrization::Orbital::d,
					Parametrization::Bond::Delta
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
			return (n*n - (l*l + m*m)/2.)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::s,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			);
		case Orbital::x:
			return -l*(n*n - (l*l + m*m)/2.)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			) + sqrt(3.)*l*n*n*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Pi
			);
		case Orbital::y:
			return -m*(n*n - (l*l + m*m)/2.)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			) + sqrt(3.)*m*n*n*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Pi
			);
		case Orbital::z:
			return -n*(n*n - (l*l + m*m)/2.)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			) - sqrt(3.)*n*(l*l + m*m)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::p,
				Parametrization::Orbital::d,
				Parametrization::Bond::Pi
			);
		case Orbital::xy:
			return sqrt(3.)*(
				l*m*(n*n - (l*l + m*m)/2.)*parametrization->getParameter(
					distance,
					Parametrization::Orbital::d,
					Parametrization::Orbital::d,
					Parametrization::Bond::Sigma
				) - 2*l*m*n*n*parametrization->getParameter(
					distance,
					Parametrization::Orbital::d,
					Parametrization::Orbital::d,
					Parametrization::Bond::Pi
				) + (l*m*(1 + n*n)/2.)*parametrization->getParameter(
					distance,
					Parametrization::Orbital::d,
					Parametrization::Orbital::d,
					Parametrization::Bond::Delta
				)
			);
		case Orbital::yz:
			return sqrt(3.)*(
				m*n*(n*n - (l*l + m*m)/2.)*parametrization->getParameter(
					distance,
					Parametrization::Orbital::d,
					Parametrization::Orbital::d,
					Parametrization::Bond::Sigma
				) + m*n*(l*l + m*m - n*n)*parametrization->getParameter(
					distance,
					Parametrization::Orbital::d,
					Parametrization::Orbital::d,
					Parametrization::Bond::Pi
				) - (m*n*(l*l + m*m)/2.)*parametrization->getParameter(
					distance,
					Parametrization::Orbital::d,
					Parametrization::Orbital::d,
					Parametrization::Bond::Delta
				)
			);
		case Orbital::zx:
			return sqrt(3.)*(
				l*n*(n*n - (l*l + m*m)/2.)*parametrization->getParameter(
					distance,
					Parametrization::Orbital::d,
					Parametrization::Orbital::d,
					Parametrization::Bond::Sigma
				) + l*n*(l*l + m*m - n*n)*parametrization->getParameter(
					distance,
					Parametrization::Orbital::d,
					Parametrization::Orbital::d,
					Parametrization::Bond::Pi
				) - (l*n*(l*l + m*m)/2.)*parametrization->getParameter(
					distance,
					Parametrization::Orbital::d,
					Parametrization::Orbital::d,
					Parametrization::Bond::Delta
				)
			);
		case Orbital::x2my2:
			return sqrt(3.)*(
				(l*l - m*m)*(n*n - (l*l + m*m)/2.)*parametrization->getParameter(
					distance,
					Parametrization::Orbital::d,
					Parametrization::Orbital::d,
					Parametrization::Bond::Sigma
				)/2. + n*n*(m*m - l*l)*parametrization->getParameter(
					distance,
					Parametrization::Orbital::d,
					Parametrization::Orbital::d,
					Parametrization::Bond::Pi
				) + ((1 + n*n)*(l*l - m*m)/4.)*parametrization->getParameter(
					distance,
					Parametrization::Orbital::d,
					Parametrization::Orbital::d,
					Parametrization::Bond::Delta
				)
			);
		case Orbital::z2mr2:
			return pow(n*n - (l*l + m*m)/2., 2)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Sigma
			) + 3*n*n*(l*l + m*m)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Pi
			) + (3./4.)*pow(l*l +m*m, 2)*parametrization->getParameter(
				distance,
				Parametrization::Orbital::d,
				Parametrization::Orbital::d,
				Parametrization::Bond::Delta
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

SlaterKosterState::Parametrization::~Parametrization(){
}

};	//End of namespace TBTK
