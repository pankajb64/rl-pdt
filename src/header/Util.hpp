//
// Created by pankaj on 12/20/17.
//

#pragma once

#include <stdafx.hpp>
using namespace std;
class Util
{
public:
    //Tokenize a string based on delimiter
    //Ref - https://stackoverflow.com/a/37454181/1026535
    static vector<int> split(const string& str, const string& delim)
    {
        vector<int> tokens;
        size_t prev = 0, pos = 0;
        do
        {
            pos = str.find(delim, prev);
            if (pos == string::npos) pos = str.length();
            string token = str.substr(prev, pos-prev);
            if (!token.empty()) tokens.push_back(stoi(token));
            prev = pos + delim.length();
        }
        while (pos < str.length() && prev < str.length());
        //cout << "tokens.size() " << tokens.size() << endl;
        return tokens;
    }
};