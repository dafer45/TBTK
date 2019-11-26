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

/** @file IndexBasedHoppingAMplitudeFilter.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/IndexBasedHoppingAmplitudeFilter.h"

namespace TBTK{

IndexBasedHoppingAmplitudeFilter::IndexBasedHoppingAmplitudeFilter(
	const AbstractIndexFilter &indexFilter
){
	this->indexFilter = indexFilter.clone();
}

IndexBasedHoppingAmplitudeFilter::IndexBasedHoppingAmplitudeFilter(
	const IndexBasedHoppingAmplitudeFilter &indexBasedHoppingAmplitudeFilter
){
	if(indexBasedHoppingAmplitudeFilter.indexFilter == nullptr){
		indexFilter = nullptr;
	}
	else{
		indexFilter
			= indexBasedHoppingAmplitudeFilter.indexFilter->clone();
	}
}

IndexBasedHoppingAmplitudeFilter::IndexBasedHoppingAmplitudeFilter(
	IndexBasedHoppingAmplitudeFilter &&indexBasedHoppingAmplitudeFilter
){
	this->indexFilter = indexBasedHoppingAmplitudeFilter.indexFilter;
	indexBasedHoppingAmplitudeFilter.indexFilter = nullptr;
}

IndexBasedHoppingAmplitudeFilter::~IndexBasedHoppingAmplitudeFilter(){
	if(indexFilter != nullptr)
		delete indexFilter;
}

IndexBasedHoppingAmplitudeFilter& IndexBasedHoppingAmplitudeFilter::operator=(
	const IndexBasedHoppingAmplitudeFilter &rhs
){
	if(this != &rhs){
		if(indexFilter != nullptr)
			delete indexFilter;

		if(rhs.indexFilter == nullptr)
			indexFilter = nullptr;
		else
			indexFilter = rhs.indexFilter->clone();
	}

	return *this;
}

IndexBasedHoppingAmplitudeFilter& IndexBasedHoppingAmplitudeFilter::operator=(
	IndexBasedHoppingAmplitudeFilter &&rhs
){
	if(this != &rhs){
		if(indexFilter != nullptr)
			delete indexFilter;

		indexFilter = rhs.indexFilter->clone();
		rhs.indexFilter = nullptr;
	}

	return *this;
}

IndexBasedHoppingAmplitudeFilter* IndexBasedHoppingAmplitudeFilter::clone() const{
	return new IndexBasedHoppingAmplitudeFilter(*this);
}

bool IndexBasedHoppingAmplitudeFilter::isIncluded(
	const HoppingAmplitude &hoppingAmplitude
) const{
	return (
		indexFilter->isIncluded(hoppingAmplitude.getToIndex())
		&& indexFilter->isIncluded(hoppingAmplitude.getFromIndex())
	);
}

};	//End of namespace TBTK
