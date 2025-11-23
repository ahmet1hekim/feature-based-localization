#include <SFML/Graphics.hpp>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>

cv::Mat sfml2opencv(const sf::Image &img)
{
    // get size from image
    cv::Size size(img.getSize().x, img.getSize().y);
    // create a mat on image memory
    cv::Mat mat(size, CV_8UC4, (void *)img.getPixelsPtr(), cv::Mat::AUTO_STEP);
    // make SFML RGBA to OpenCV BGRA
    cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);
    return mat.clone();
}

int main()
{
    // how big are the tiles
    u_int16_t tile_res_x = 960;
    u_int16_t tile_res_y = 540;
    // how many tiles are there in the original image
    uint16_t background_tile_count_x = 4;
    uint16_t background_tile_count_y = 4;
    // how many tiles are there in a window
    uint16_t window_tile_count_x = 2;
    uint16_t window_tile_count_y = 2;
    // original image's res
    u_int16_t background_res_x = tile_res_x * background_tile_count_x;
    u_int16_t background_res_y = tile_res_y * background_tile_count_y;

    // current windpw's res
    u_int16_t window_res_x = tile_res_x * window_tile_count_x;
    u_int16_t window_res_y = tile_res_y * window_tile_count_y;

    // current window sized block on the texture - used in calculating positions on the original texture
    u_int16_t curr_window_x = 0;
    u_int16_t curr_window_y = 0;

    u_int16_t max_window_x = background_res_x / window_res_x;
    u_int16_t max_window_y = background_res_y / window_res_y;

    // creating the window and setting vsync so my pc doesnt die
    sf::RenderWindow window(sf::VideoMode({window_res_x, window_res_y}), "SFML works!", sf::Style::None, sf::State::Windowed);
    window.setVerticalSyncEnabled(true); // call it once after creating the window
    float dot_r = 10.f;
    // the little dot on the screen
    // sf::CircleShape shape(dot_r);

    float rect_h = 30.f, rect_w = 20.f;

    sf::RectangleShape shape({rect_w, rect_h});
    shape.setFillColor(sf::Color::Red);
    // centering its control point
    shape.setOrigin({rect_w / 2, rect_h / 2});
    // centering it on the window
    shape.setPosition({window_res_x / 2, window_res_y / 2});

    // getting the image and creating the texture and the sprites
    sf::FileInputStream bg_stream(std::string(ASSETS_DIR) + "/dag.jpg");
    sf::Texture bg_texture(bg_stream);
    sf::Sprite bg_sprite(bg_texture);

    // setting the sprite's starting point to texture's 0,0 and its size to the window size
    bg_sprite.setTextureRect(sf::IntRect({0, 0}, {window_res_x, window_res_y}));

    // speed of the little dot
    auto speed_x = 3.5f, speed_y = 3.5f;
    // space changed by the little dot
    auto tomove_x = 0.0f, tomove_y = 0.0f;
    // offset values for checks
    auto offset_x = 2.0f, offset_y = 2.0f;

    // second window for debug
    sf::RenderWindow window2(sf::VideoMode({tile_res_x, tile_res_y}), "SFML works!", sf::Style::None, sf::State::Windowed);
    window2.setVerticalSyncEnabled(true); // call it once after creating the window

    while (window.isOpen())
    {
        // bg_sprite.setTextureRect(sf::IntRect({shape.getPosition().x, shape.getPosition().y}, {window_res_x, window_res_y}));

        // std::cout << std::string(ASSETS_DIR) + "/dag.jpg";

        // HANDLE KEYBOARD EVENTS
        while (const std::optional event = window.pollEvent())
        {
            if (event->is<sf::Event::Closed>())
            {
                window.close();
            }
        }

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Left))
        {
            // std::cout << "move left";
            // tomove_x = -speed_x;
            shape.rotate(sf::degrees(-2.5f));
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Right))
        {
            // std::cout << "move right ";
            // tomove_x = speed_x;
            shape.rotate(sf::degrees(2.5f));
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Down))
        {
            // std::cout << "move left";
            tomove_y = speed_y;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Up))
        {
            // std::cout << "move right ";
            tomove_y = -speed_y;
        }
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Space))
        {
            // std::cout << "move right ";
            speed_y += 2.5f;
            speed_x += 2.5f;
            offset_x = speed_x * 2, offset_y = speed_y * 2;
        }

        // HANDLE LITTLE DOTS MOVEMENT
        // if (shape.getPosition().x + tomove_x < window_res_x+offset_x && shape.getPosition().x + tomove_x >-offset_x && shape.getPosition().y + tomove_y < window_res_y +offset_y && shape.getPosition().y + tomove_y >-offset_y)
        // {
        //     float rad = shape.getRotation().asDegrees() * 3.14159265f / 180.f;
        //     shape.move({std::sin(rad) * tomove_y, -std::cos(rad) * tomove_y});
        // }

        float rad = shape.getRotation().asDegrees() * 3.14159265f / 180.f;
        sf::Vector2f moveVec(std::sin(rad) * tomove_y, -std::cos(rad) * tomove_y);

        // Compute future position
        sf::Vector2f futurePos = shape.getPosition() + moveVec;

        // Get screen bounds (assuming (0,0) top-left)
        if (futurePos.x > -offset_x && futurePos.x < window_res_x + offset_x &&
            futurePos.y > -offset_y && futurePos.y < window_res_y + offset_y)
        {
            shape.move(moveVec);
        }

        // HANDLE TILE MOVEMENT

        if (shape.getPosition().x >= window_res_x - offset_x && curr_window_x < (max_window_x - 1))

        {
            std::cout << "big jump" << std::endl;
            curr_window_x++;

            bg_sprite.setTextureRect(sf::IntRect({tile_res_x * curr_window_x, tile_res_y * curr_window_y}, {window_res_x, window_res_y}));

            shape.setPosition({offset_x + 20, shape.getPosition().y});
        }

        else if (shape.getPosition().x <= offset_x && curr_window_x > 0)

        {
            std::cout << "big jump" << std::endl;
            curr_window_x--;

            bg_sprite.setTextureRect(sf::IntRect({tile_res_x * curr_window_x, tile_res_y * curr_window_y}, {window_res_x, window_res_y}));

            shape.setPosition({window_res_x - offset_x - 20, shape.getPosition().y});
        }

        if (shape.getPosition().y >= window_res_y - offset_y && curr_window_y < (max_window_y - 1))
        {
            std::cout << "big jump" << std::endl;
            curr_window_y++;

            bg_sprite.setTextureRect(sf::IntRect({tile_res_x * curr_window_x, tile_res_y * curr_window_y}, {window_res_x, window_res_y}));

            shape.setPosition({shape.getPosition().x, offset_y + 20});
        }

        else if (shape.getPosition().y <= offset_y && curr_window_y > 0)
        {
            std::cout << "big jump" << std::endl;
            curr_window_y--;

            bg_sprite.setTextureRect(sf::IntRect({tile_res_x * curr_window_x, tile_res_y * curr_window_y}, {window_res_x, window_res_y}));

            shape.setPosition({shape.getPosition().x, window_res_y - offset_y - 20});
        }

        tomove_x = 0.0f;
        tomove_y = 0.0f;

        // std::cout << shape.getPosition().x << " " << shape.getPosition().y << std::endl;
        // std::cout << shape.getPosition().x << " " << shape.getPosition().y << std::endl;

        // std::cout << max_window_x << " " << max_window_y << " " <<curr_window_x << " " << curr_window_y << std::endl;

        window.draw(bg_sprite);
        window.draw(shape);
        window.display();

        sf::View view;
        view.setCenter(shape.getPosition());
        view.setSize({tile_res_x, tile_res_y});
        view.setRotation(shape.getRotation());
        window2.setView(view);
        window2.clear();
        window2.draw(bg_sprite);
        window2.draw(shape);
        // window2.display();

        // get the image
        sf::Texture windowTexture({window2.getSize().x, window2.getSize().y});
        windowTexture.update(window2);
        sf::Image croppedImage = windowTexture.copyToImage();
        cv::imshow("drone_view", sfml2opencv(croppedImage));
        cv::waitKey(1);
    }
}
