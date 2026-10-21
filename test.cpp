
#include "imgui.h"
#include "implot.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

#include <GL/gl.h>
#include <GLFW/glfw3.h>

#include <cmath>
#include <vector>

int main()
{
    if (!glfwInit())
        return 1;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    GLFWwindow* window = glfwCreateWindow(
        1000, 700, "ImPlot Test", nullptr, nullptr
    );

    if (!window) {
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    const int count = 1000;
    std::vector<double> x(count);
    std::vector<double> y(count);

    for (int i = 0; i < count; i++)
        x[i] = i * 0.01;

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        double time = glfwGetTime();

        for (int i = 0; i < count; i++)
            y[i] = std::sin(2.0 * 3.14159265 * (x[i] + time));

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Live Waveform");

        ImGui::Text("Sine wave test");
        ImGui::Text("Time: %.2f seconds", time);

        if (ImPlot::BeginPlot("Signal"))
        {
            ImPlot::SetupAxes("Time", "Amplitude");
            ImPlot::PlotLine("Wave", x.data(), y.data(), count);
            ImPlot::EndPlot();
        }

        ImGui::End();

        ImGui::Render();

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        glViewport(0, 0, width, height);
        glClearColor(0.10f, 0.10f, 0.12f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();

    ImPlot::DestroyContext();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}

