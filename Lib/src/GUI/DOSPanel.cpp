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

/** @file DOSPanel.cpp
 *
 *  @author Kristofer Björnson
 */

#include "TBTK/Plot/Plotter.h"
#include "TBTK/DOSPanel.h"

#include <wx/gbsizer.h>

using namespace std;

namespace TBTK{

using namespace Plot;

const wxWindowID DOSPanel::GAUSSIAN_SMOOTHING_ID = wxWindow::NewControlId();
const wxWindowID DOSPanel::MIN_ID = wxWindow::NewControlId();
const wxWindowID DOSPanel::MAX_ID = wxWindow::NewControlId();

DOSPanel::DOSPanel(
	wxWindow *parent
) :
	wxPanel(parent),
	controlPanel(this)
{
	dos = nullptr;

	wxSizer *sizer = new wxBoxSizer(wxHORIZONTAL);

	resultPanel = new ImagePanel(this);
	sizer->Add(resultPanel, 1, wxEXPAND);

	sizer->Add(&controlPanel, 0, 0);

	SetSizer(sizer);

	controlPanel.Bind(wxEVT_TEXT_ENTER, &DOSPanel::onChangeEvent, this);
}

DOSPanel::~DOSPanel(){
	if(dos != nullptr)
		delete dos;

	delete resultPanel;
}

void DOSPanel::setDOS(const Property::DOS &dos){
	if(this->dos != nullptr)
		delete this->dos;

	this->dos = new Property::DOS(dos);

	updatePlot();
}

BEGIN_EVENT_TABLE(DOSPanel, wxPanel)

EVT_PAINT(DOSPanel::onPaintEvent)
EVT_SIZE(DOSPanel::onSizeEvent)

END_EVENT_TABLE()

void DOSPanel::onPaintEvent(wxPaintEvent &event){
}

void DOSPanel::onSizeEvent(wxSizeEvent &event){
	Layout();
	updatePlot();
	Refresh();
	event.Skip();
}

void DOSPanel::onChangeEvent(wxCommandEvent &event){
	updatePlot();
	Refresh();
	event.Skip();
}

DOSPanel::ControlPanel::ControlPanel(
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
	)
{
	wxSizer *sizer = new wxBoxSizer(wxVERTICAL);
	sizer->Add(&gaussianSmoothingLabel, 1, wxEXPAND);
	sizer->Add(&gaussianSmoothingTextBox, 1, wxEXPAND);
	sizer->Add(&minLabel, 1, wxEXPAND);
	sizer->Add(&minTextBox, 1, wxEXPAND);
	sizer->Add(&maxLabel, 1, wxEXPAND);
	sizer->Add(&maxTextBox, 1, wxEXPAND);
	SetSizer(sizer);

	min = -10;
	max = 10;
}

void DOSPanel::updatePlot(){
	if(dos != nullptr){
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
			*dos,
			controlPanel.getGaussianSmoothing(),
			2*dos->getResolution()+1
		);

		resultPanel->setImage(plotter.getCanvas());
	}
}

DOSPanel::ControlPanel::~ControlPanel(){
}

BEGIN_EVENT_TABLE(DOSPanel::ControlPanel, wxPanel)

EVT_TEXT_ENTER(
	GAUSSIAN_SMOOTHING_ID,
	DOSPanel::ControlPanel::onGaussianSmoothingChanged
)

EVT_TEXT(
	GAUSSIAN_SMOOTHING_ID,
	DOSPanel::ControlPanel::onGaussianSmoothingChanged
)

EVT_TEXT_ENTER(
	MIN_ID,
	DOSPanel::ControlPanel::onMinChanged
)

EVT_TEXT(
	MIN_ID,
	DOSPanel::ControlPanel::onMinChanged
)

EVT_TEXT_ENTER(
	MAX_ID,
	DOSPanel::ControlPanel::onMaxChanged
)

EVT_TEXT(
	MAX_ID,
	DOSPanel::ControlPanel::onMaxChanged
)

END_EVENT_TABLE()

void DOSPanel::ControlPanel::onGaussianSmoothingChanged(wxCommandEvent &event){
	gaussianSmoothing = atof(event.GetString());
//	((DOSPanel*)GetParent())->redraw();
}

void DOSPanel::ControlPanel::onMinChanged(wxCommandEvent &event){
	min = atof(event.GetString());
//	((DOSPanel*)GetParent())->redraw();
}

void DOSPanel::ControlPanel::onMaxChanged(wxCommandEvent &event){
	max = atof(event.GetString());
//	((DOSPanel*)GetParent())->redraw();
}

};	//End of namespace TBTK
