#include "TBTK/InteractionAmplitude.h"

using namespace std;

namespace TBTK{

InteractionAmplitude::InteractionAmplitude(){
	amplitudeCallback = nullptr;
}

InteractionAmplitude::InteractionAmplitude(
	complex<double> amplitude,
	initializer_list<Index> creationOperatorIndices,
	initializer_list<Index> annihilationOperatorIndices
){
	this->amplitude = amplitude;
	this->amplitudeCallback = NULL;

	for(unsigned int n = 0; n < creationOperatorIndices.size(); n++)
		this->creationOperatorIndices.push_back(*(creationOperatorIndices.begin()+n));

	for(unsigned int n = 0; n < annihilationOperatorIndices.size(); n++)
		this->annihilationOperatorIndices.push_back(*(annihilationOperatorIndices.begin()+n));
}

InteractionAmplitude::InteractionAmplitude(
	complex<double> (*amplitudeCallback)(const std::vector<Index>&, const std::vector<Index>&),
	initializer_list<Index> creationOperatorIndices,
	initializer_list<Index> annihilationOperatorIndices
){
	this->amplitudeCallback = amplitudeCallback;

	for(unsigned int n = 0; n < creationOperatorIndices.size(); n++)
		this->creationOperatorIndices.push_back(*(creationOperatorIndices.begin()+n));

	for(unsigned int n = 0; n < annihilationOperatorIndices.size(); n++)
		this->annihilationOperatorIndices.push_back(*(annihilationOperatorIndices.begin()+n));
}

InteractionAmplitude::InteractionAmplitude(const InteractionAmplitude &ia){
	this->amplitude = ia.amplitude;
	this->amplitudeCallback = ia.amplitudeCallback;

	for(unsigned int n = 0; n < ia.creationOperatorIndices.size(); n++)
		this->creationOperatorIndices.push_back(ia.creationOperatorIndices.at(n));

	for(unsigned int n = 0; n < ia.annihilationOperatorIndices.size(); n++)
		this->annihilationOperatorIndices.push_back(ia.annihilationOperatorIndices.at(n));
}

InteractionAmplitude::~InteractionAmplitude(){
}

};	//End of namespace TBTK
