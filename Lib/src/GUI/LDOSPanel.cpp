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

/** @file LDOSPanel.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/IndexException.h"
#include "TBTK/Plot/Plotter.h"
#include "TBTK/Smooth.h"
#include "TBTK/LDOSPanel.h"

#include <wx/gbsizer.h>

using namespace std;

namespace TBTK{

using namespace Plot;

const wxWindowID LDOSPanel::GAUSSIAN_SMOOTHING_ID = wxWindow::NewControlId();
const wxWindowID LDOSPanel::MIN_ID = wxWindow::NewControlId();
const wxWindowID LDOSPanel::MAX_ID = wxWindow::NewControlId();

LDOSPanel::LDOSPanel(
	wxWindow *parent
) :
	wxPanel(parent),
	controlPanel(this)
{
	ldos = nullptr;

	wxSizer *sizer = new wxBoxSizer(wxHORIZONTAL);

	resultPanel = new ImagePanel(this);
	sizer->Add(resultPanel, 1, wxEXPAND);

	sizer->Add(&controlPanel, 0, 0);

	SetSizer(sizer);

	controlPanel.Bind(wxEVT_TEXT_ENTER, &LDOSPanel::onChange, this);
}

LDOSPanel::~LDOSPanel(){
	if(ldos != nullptr)
		delete ldos;

	delete resultPanel;
}

void LDOSPanel::setLDOS(const Property::LDOS &ldos){
	if(this->ldos != nullptr)
		delete this->ldos;

	this->ldos = new Property::LDOS(ldos);

	updatePlot();
}

BEGIN_EVENT_TABLE(LDOSPanel, wxPanel)

EVT_PAINT(LDOSPanel::onPaintEvent)
EVT_SIZE(LDOSPanel::onSizeEvent)

END_EVENT_TABLE()

void LDOSPanel::onPaintEvent(wxPaintEvent &event){
}

void LDOSPanel::onSizeEvent(wxSizeEvent &event){
	Layout();
	updatePlot();
	Refresh();
	event.Skip();
}

void LDOSPanel::onChange(wxCommandEvent &event){
	updatePlot();
	event.Skip();
	Refresh();
}

LDOSPanel::ControlPanel::ControlPanel(
	wxWindow *parent
) :
	wxPanel(parent),
	gaussianSmoothingLabel(
		this,
		-1,
		"Gaussian smoothing (sigma)"
	),
	gaussianSmoothingTextBox(
		this,
		GAUSSIAN_SMOOTHING_ID,
		"0.1",
		wxDefaultPosition,
		wxDefaultSize,
		wxTE_PROCESS_ENTER | wxTE_PROCESS_TAB,
		wxTextValidator(wxFILTER_NUMERIC),
		wxTextCtrlNameStr
	),
	minLabel(
		this,
		-1,
		"Min energy"
	),
	minTextBox(
		this,
		MIN_ID,
		"-10",
		wxDefaultPosition,
		wxDefaultSize,
		wxTE_PROCESS_ENTER | wxTE_PROCESS_TAB,
		wxTextValidator(wxFILTER_NUMERIC),
		wxTextCtrlNameStr
	),
	maxLabel(
		this,
		-1,
		"Max energy"
	),
	maxTextBox(
		this,
		MAX_ID,
		"10",
		wxDefaultPosition,
		wxDefaultSize,
		wxTE_PROCESS_ENTER | wxTE_PROCESS_TAB,
		wxTextValidator(wxFILTER_NUMERIC),
		wxTextCtrlNameStr
	),
	indexPanel(
		this
	)
{
	wxSizer *sizer = new wxBoxSizer(wxVERTICAL);
	sizer->Add(&gaussianSmoothingLabel, 0, wxEXPAND);
	sizer->Add(&gaussianSmoothingTextBox, 0, wxEXPAND);
	sizer->Add(&minLabel, 0, wxEXPAND);
	sizer->Add(&minTextBox, 0, wxEXPAND);
	sizer->Add(&maxLabel, 0, wxEXPAND);
	sizer->Add(&maxTextBox, 0, wxEXPAND);
	sizer->Add(&indexPanel, 0, wxEXPAND);
	SetSizer(sizer);
}

void LDOSPanel::updatePlot(){
	if(ldos != nullptr){
		try{
			vector<double> axis;
			vector<double> data;
			for(int n = 0; n < (int)ldos->getResolution(); n++){
				axis.push_back(
					ldos->getLowerBound()
					+ (ldos->getUpperBound() - ldos->getLowerBound())/ldos->getResolution()*n
				);
				data.push_back(
					(*ldos)(controlPanel.getIndex(), n)
				);
			}

			int width, height;
			resultPanel->GetSize(&width, &height);

			Plotter plotter;
			cv::Mat canvas;
			plotter.setCanvas(canvas);
			plotter.setWidth(width);
			plotter.setHeight(height);
			plotter.setBoundsX(
				controlPanel.getMin(),
				controlPanel.getMax()
			);
			plotter.plot(
				axis,
				Smooth::gaussian(
					data,
					controlPanel.getGaussianSmoothing(),
					2*ldos->getResolution()+1
				)
			);

			resultPanel->setImage(plotter.getCanvas());
		}
		catch(IndexException &e){
		}
	}
}

LDOSPanel::ControlPanel::~ControlPanel(){
}

BEGIN_EVENT_TABLE(LDOSPanel::ControlPanel, wxPanel)

EVT_TEXT_ENTER(
	GAUSSIAN_SMOOTHING_ID,
	LDOSPanel::ControlPanel::onGaussianSmoothingChanged
)

EVT_TEXT(
	GAUSSIAN_SMOOTHING_ID,
	LDOSPanel::ControlPanel::onGaussianSmoothingChanged
)

EVT_TEXT_ENTER(
	MIN_ID,
	LDOSPanel::ControlPanel::onMinChanged
)

EVT_TEXT(
	MIN_ID,
	LDOSPanel::ControlPanel::onMinChanged
)

EVT_TEXT_ENTER(
	MAX_ID,
	LDOSPanel::ControlPanel::onMaxChanged
)

EVT_TEXT(
	MAX_ID,
	LDOSPanel::ControlPanel::onMaxChanged
)

END_EVENT_TABLE()

void LDOSPanel::ControlPanel::onGaussianSmoothingChanged(wxCommandEvent &event){
	gaussianSmoothing = atof(event.GetString());
}

void LDOSPanel::ControlPanel::onMinChanged(wxCommandEvent &event){
	min = atof(event.GetString());
}

void LDOSPanel::ControlPanel::onMaxChanged(wxCommandEvent &event){
	max = atof(event.GetString());
}

};	//End of namespace TBTK
