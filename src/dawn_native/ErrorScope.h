// Copyright 2019 The Dawn Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DAWNNATIVE_ERRORSCOPE_H_
#define DAWNNATIVE_ERRORSCOPE_H_

#include "dawn_native/dawn_platform.h"

#include "dawn_native/RefCounted.h"

#include <string>

namespace dawn_native {

    // Errors can be recorded into an ErrorScope by calling |HandleError|.
    // Because an error scope should not resolve until contained
    // commands are complete, calling the callback is deferred until it is destructed.
    // In-flight commands or asynchronous events should hold a reference to the
    // ErrorScope for their duration.
    //
    // Because parent ErrorScopes should not resolve before child ErrorScopes,
    // ErrorScopes hold a reference to their parent.
    //
    // To simplify ErrorHandling, there is a sentinel root error scope which has
    // no parent. All uncaptured errors are handled by the root error scope. Its
    // callback is called immediately once it encounters an error.
    class ErrorScope : public RefCounted {
      public:
        ErrorScope();  // Constructor for the root error scope.
        ErrorScope(dawn::ErrorFilter errorFilter, ErrorScope* parent);
        ~ErrorScope();

        void SetCallback(dawn::ErrorCallback callback, void* userdata);
        ErrorScope* GetParent();

        void HandleError(dawn::ErrorType type, const char* message);

        void Destroy();

      private:
        bool IsRoot() const;
        static void HandleErrorImpl(ErrorScope* scope, dawn::ErrorType type, const char* message);

        dawn::ErrorFilter mErrorFilter = dawn::ErrorFilter::None;
        Ref<ErrorScope> mParent = nullptr;

        dawn::ErrorCallback mCallback = nullptr;
        void* mUserdata = nullptr;

        dawn::ErrorType mErrorType = dawn::ErrorType::NoError;
        std::string mErrorMessage = "";
    };

}  // namespace dawn_native

#endif  // DAWNNATIVE_ERRORSCOPE_H_
