document.addEventListener("DOMContentLoaded", function () {
  var plotDataElement = document.getElementById("plot-data");
  var plotContainerElement = document.getElementById("affinitree-plot");
  var radialChartContainerElement = document.getElementById("radial-chart-container");

  if (!plotDataElement || !plotContainerElement || !radialChartContainerElement) {
    console.error("Required elements #plot-data, #affinitree-plot, or #radial-chart-container are missing.");
    return;
  }

  var plotData;
  try {
    plotData = JSON.parse(plotDataElement.textContent);
  } catch (error) {
    console.error("Could not parse Plotly data JSON from #plot-data:", error);
    return;
  }

  var layout = plotData.layout || {};
  layout.dragmode = false;
  layout.hovermode = "closest";

  var configuration = {
    displayModeBar: false,
    responsive: true,
    scrollZoom: false
  };

  Plotly.newPlot(plotContainerElement, plotData.data, layout, configuration).then(function () {
    plotContainerElement.style.cursor = "pointer";

    plotContainerElement.on("plotly_hover", function () {
      plotContainerElement.style.cursor = "pointer";
    });

    plotContainerElement.on("plotly_unhover", function () {
      plotContainerElement.style.cursor = "pointer";
    });

    plotContainerElement.on("plotly_click", function (eventData) {
      if (!eventData || !eventData.points || eventData.points.length === 0) {
        return;
      }

      var point = eventData.points[0];
      var customDataValue = point.customdata;

      if (!customDataValue) {
        console.warn("Clicked point has no customdata image attached.");
        return;
      }

      var imageSource;
      if (typeof customDataValue === "string" && customDataValue.startsWith("data:image")) {
        imageSource = customDataValue;
      } else if (typeof customDataValue === "string") {
        imageSource = "data:image/png;base64," + customDataValue;
      } else {
        console.warn("Unexpected customdata format for radial chart image:", customDataValue);
        return;
      }

      while (radialChartContainerElement.firstChild) {
        radialChartContainerElement.removeChild(radialChartContainerElement.firstChild);
      }

      var imageElement = document.createElement("img");
      imageElement.src = imageSource;
      imageElement.alt = "Radial temperament and character trait profile";
      imageElement.style.maxWidth = "100%";
      imageElement.style.height = "auto";
      imageElement.style.display = "block";

      radialChartContainerElement.appendChild(imageElement);

      var selectedPointsArray = new Array(plotData.data.length).fill(null);
      Plotly.restyle(plotContainerElement, { selectedpoints: selectedPointsArray });
    });

    window.addEventListener("resize", function () {
      Plotly.Plots.resize(plotContainerElement);
    });
  });
});