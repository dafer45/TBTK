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

/** @file HoppingAmplitudePanel.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/HoppingAmplitudePanel.h"

#include <sstream>

#include <wx/gbsizer.h>

using namespace std;

namespace TBTK{

const wxWindowID HoppingAmplitudePanel::AMPLITUDE_ID = wxWindow::NewControlId();
const wxWindowID HoppingAmplitudePanel::TO_INDEX_ID = wxWindow::NewControlId();
const wxWindowID HoppingAmplitudePanel::FROM_INDEX_ID = wxWindow::NewControlId();

HoppingAmplitudePanel::HoppingAmplitudePanel(
	wxWindow *parent
) :
	wxPanel(parent),
	hoppingAmplitude(0, {}, {}),
	hoppingAmplitudeLabel(
		this,
		wxID_ANY,
		"HoppingAmplitude"
	),
	amplitudeTextBox(
		this,
		AMPLITUDE_ID,
		"(1, 0)",
		wxDefaultPosition,
		wxDefaultSize,
		wxTE_PROCESS_ENTER | wxTE_PROCESS_TAB,
		wxDefaultValidator,
		wxTextCtrlNameStr
	),
	toIndexPanel(this, TO_INDEX_ID),
	fromIndexPanel(this, FROM_INDEX_ID)
{
	wxSizer *sizer = new wxBoxSizer(wxVERTICAL);
	sizer->Add(&hoppingAmplitudeLabel, 0, wxEXPAND);
	sizer->Add(&amplitudeTextBox, 0, wxEXPAND);
	sizer->Add(&toIndexPanel, 0, wxEXPAND);
	sizer->Add(&fromIndexPanel, 0, wxEXPAND);
	SetSizer(sizer);

	toIndexPanel.Bind(
		wxEVT_TEXT_ENTER,
		&HoppingAmplitudePanel::onIndexChanged,
		this
	);
	fromIndexPanel.Bind(
		wxEVT_TEXT_ENTER,
		&HoppingAmplitudePanel::onIndexChanged,
		this
	);
}

HoppingAmplitudePanel::~HoppingAmplitudePanel(){
}

void HoppingAmplitudePanel::setHoppingAmplitude(const HoppingAmplitude &hoppingAmplitude){
	this->hoppingAmplitude = hoppingAmplitude;
	Refresh();
}

BEGIN_EVENT_TABLE(HoppingAmplitudePanel, wxPanel)

EVT_PAINT(HoppingAmplitudePanel::onPaintEvent)
EVT_SIZE(HoppingAmplitudePanel::onSizeEvent)
EVT_TEXT_ENTER(AMPLITUDE_ID, HoppingAmplitudePanel::onAmplitudeChanged)
EVT_TEXT(AMPLITUDE_ID, HoppingAmplitudePanel::onAmplitudeChanged)

END_EVENT_TABLE()

void HoppingAmplitudePanel::onPaintEvent(wxPaintEvent &event){
}

void HoppingAmplitudePanel::onSizeEvent(wxSizeEvent &event){
	Layout();
	Refresh();
	event.Skip();
}

void HoppingAmplitudePanel::onAmplitudeChanged(wxCommandEvent &event){
	stringstream ss(string(event.GetString()));
	complex<double> amplitude;
	ss >> amplitude;
	hoppingAmplitude = HoppingAmplitude(
		amplitude,
		toIndexPanel.getIndex(),
		fromIndexPanel.getIndex()
	);
	event.Skip();
}

void HoppingAmplitudePanel::onIndexChanged(wxCommandEvent &event){
	hoppingAmplitude = HoppingAmplitude(
		hoppingAmplitude.getAmplitude(),
		toIndexPanel.getIndex(),
		fromIndexPanel.getIndex()
	);
	event.Skip();
}

};	//End of namespace TBTK
