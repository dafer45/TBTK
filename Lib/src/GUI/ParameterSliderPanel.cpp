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

/** @file ParameterSliderPanel.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/ParameterSliderPanel.h"
#include "TBTK/Streams.h"
#include "TBTK/TBTKMacros.h"

using namespace std;

namespace TBTK{

const wxWindowID ParameterSliderPanel::SLIDER_ID = wxWindow::NewControlId();

ParameterSliderPanel::ParameterSliderPanel(
	wxWindow *parent,
	wxWindowID id,
	const string &parameterName,
	double lowerBound,
	double upperBound,
	unsigned int numTicks,
	double value
) :
	wxPanel(parent, id),
	label(
		this,
		wxID_ANY,
		parameterName
	),
	slider(
		this,
		SLIDER_ID,
		0,
		0,
		numTicks-1,
		wxDefaultPosition,
		wxDefaultSize,
		wxSL_HORIZONTAL,
		wxDefaultValidator,
		wxSliderNameStr
	),
	valuePanel(
		this,
		wxID_ANY,
		to_string(value)
	)
{
	TBTKAssert(
		numTicks > 0,
		"ParameterSliderPanel::ParameterSliderPanel()",
		"numTicks must be larger than 0.",
		""
	);
	TBTKAssert(
		lowerBound < upperBound,
		"ParameterSliderPanel::ParameterSliderPanel()",
		"lowerBound must be smaller than upperBound.",
		""
	);
	TBTKAssert(
		lowerBound <= value && value <= upperBound,
		"ParameterSliderPanel::ParameterSliderPanel()",
		"value must be in the interval [lowerBound, upperBound].",
		""
	);

	this->lowerBound = lowerBound;
	this->upperBound = upperBound;
	this->numTicks = numTicks;
	this->value = value;

	wxSizer *sizer = new wxBoxSizer(wxHORIZONTAL);
	sizer->Add(&label, 1, wxEXPAND);
	sizer->Add(&slider, 1, wxEXPAND);
	sizer->Add(&valuePanel, 1, wxEXPAND);
	SetSizer(sizer);
}

ParameterSliderPanel::~ParameterSliderPanel(){
}

BEGIN_EVENT_TABLE(ParameterSliderPanel, wxPanel)

EVT_PAINT(ParameterSliderPanel::onPaintEvent)
EVT_SIZE(ParameterSliderPanel::onSizeEvent)
EVT_COMMAND_SCROLL(SLIDER_ID, ParameterSliderPanel::onSliderChange)

END_EVENT_TABLE()

void ParameterSliderPanel::onPaintEvent(wxPaintEvent &event){
}

void ParameterSliderPanel::onSizeEvent(wxSizeEvent &event){
	Layout();
	Refresh();
	event.Skip();
}

void ParameterSliderPanel::onSliderChange(wxScrollEvent &event){
	if(numTicks == 1)
		value =  lowerBound;
	else
		value = lowerBound + slider.GetValue()*(upperBound - lowerBound)/(numTicks-1);

	valuePanel.SetLabel(to_string(value));

	Refresh();
}

};	//End of namespace TBTK
