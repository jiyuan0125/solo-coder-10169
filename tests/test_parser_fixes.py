#!/usr/bin/env python

import unittest
import sys

sys.path.insert(0, ".")

from pycparser import c_parser, c_ast, c_generator
from pycparser.ast_transforms import fix_switch_cases, fix_atomic_specifiers

ParseError = c_parser.ParseError

_c_parser = c_parser.CParser()
_c_generator = c_generator.CGenerator()


def expand_decl(decl):
    typ = type(decl)

    if typ == c_ast.TypeDecl:
        return ["TypeDecl", expand_decl(decl.type)]
    elif typ == c_ast.IdentifierType:
        return ["IdentifierType", decl.names]
    elif typ == c_ast.ID:
        return ["ID", decl.name]
    elif typ in [c_ast.Struct, c_ast.Union]:
        decls = [expand_decl(d) for d in decl.decls or []]
        return [typ.__name__, decl.name, decls]
    elif typ == c_ast.Enum:
        if decl.values is None:
            values = None
        else:
            assert isinstance(decl.values, c_ast.EnumeratorList)
            values = [enum.name for enum in decl.values.enumerators]
        return ["Enum", decl.name, values]
    elif typ == c_ast.Constant:
        return ["Constant", decl.type, decl.value]
    else:
        nested = expand_decl(decl.type)

        if typ == c_ast.Decl:
            return ["Decl", decl.name, nested]
        elif typ == c_ast.ArrayDecl:
            dimval = decl.dim.value if decl.dim else ""
            return ["ArrayDecl", dimval, nested]
        elif typ == c_ast.PtrDecl:
            return ["PtrDecl", nested]
        elif typ == c_ast.FuncDecl:
            if decl.args:
                params = [expand_decl(param) for param in decl.args.params]
            else:
                params = []
            return ["FuncDecl", params, nested]


def fdef_decl(fdef):
    return expand_decl(fdef.decl)


def compare_asts(ast1, ast2):
    if type(ast1) is not type(ast2):
        return False
    elif isinstance(ast1, (list, tuple)):
        if len(ast1) != len(ast2):
            return False
        for i in range(len(ast1)):
            if not compare_asts(ast1[i], ast2[i]):
                return False
        return True
    elif isinstance(ast1, c_ast.Node):
        for attr in ast1.attr_names:
            attr1 = getattr(ast1, attr)
            attr2 = getattr(ast2, attr)
            if not compare_asts(attr1, attr2):
                return False
        children1 = ast1.children()
        children2 = ast2.children()
        if len(children1) != len(children2):
            return False
        for i in range(len(children1)):
            if not compare_asts(children1[i], children2[i]):
                return False
        return True
    else:
        return ast1 == ast2


def roundtrip_ok(src):
    ast1 = _c_parser.parse(src)
    generated = _c_generator.visit(ast1)
    ast2 = _c_parser.parse(generated)
    return compare_asts(ast1, ast2)


class TestIntegerSuffixes(unittest.TestCase):
    def test_decimal_suffixes(self):
        tests = [
            ("42", "int"),
            ("42u", "unsigned int"),
            ("42U", "unsigned int"),
            ("42l", "long int"),
            ("42L", "long int"),
            ("42ul", "unsigned long int"),
            ("42LU", "unsigned long int"),
            ("42ll", "long long int"),
            ("42LL", "long long int"),
            ("42ULL", "unsigned long long int"),
            ("42llu", "unsigned long long int"),
        ]
        for val, expected_type in tests:
            code = f"int x = {val};"
            ast = _c_parser.parse(code)
            init = ast.ext[0].init
            self.assertEqual(init.type, expected_type, f"Failed for {val}")

    def test_hex_suffixes(self):
        tests = [
            ("0xFF", "int"),
            ("0xFFu", "unsigned int"),
            ("0xFFL", "long int"),
            ("0xFFull", "unsigned long long int"),
        ]
        for val, expected_type in tests:
            code = f"int x = {val};"
            ast = _c_parser.parse(code)
            init = ast.ext[0].init
            self.assertEqual(init.type, expected_type, f"Failed for {val}")

    def test_octal_suffixes(self):
        tests = [
            ("077", "int"),
            ("077U", "unsigned int"),
            ("077ll", "long long int"),
        ]
        for val, expected_type in tests:
            code = f"int x = {val};"
            ast = _c_parser.parse(code)
            init = ast.ext[0].init
            self.assertEqual(init.type, expected_type, f"Failed for {val}")

    def test_binary_suffixes(self):
        tests = [
            ("0b1010", "int"),
            ("0b1010u", "unsigned int"),
            ("0B1010L", "long int"),
            ("0b1010ull", "unsigned long long int"),
        ]
        for val, expected_type in tests:
            code = f"int x = {val};"
            ast = _c_parser.parse(code)
            init = ast.ext[0].init
            self.assertEqual(init.type, expected_type, f"Failed for {val}")

    def test_invalid_suffixes(self):
        invalid = [
            "42uu",
            "42lll",
            "42luul",
            "42xx",
            "42uul",
            "42lllu",
        ]
        for val in invalid:
            code = f"int x = {val};"
            with self.assertRaises(ParseError, msg=f"Should have failed for {val}"):
                _c_parser.parse(code)

    def test_multichar_constant_not_affected(self):
        code = "int x = 'LL';"
        ast = _c_parser.parse(code)
        init = ast.ext[0].init
        self.assertEqual(init.type, "int")
        self.assertEqual(init.value, "'LL'")


class TestFloatSuffixes(unittest.TestCase):
    def test_valid_suffixes(self):
        tests = [
            ("1.0", "double"),
            ("1.0f", "float"),
            ("1.0F", "float"),
            ("1.0l", "long double"),
            ("1.0L", "long double"),
            ("1e10", "double"),
            ("1e10f", "float"),
            ("1E10L", "long double"),
            ("0xDE.4p1", "double"),
            ("0xDE.4p1f", "float"),
            ("0XDE.4P1L", "long double"),
        ]
        for val, expected_type in tests:
            code = f"double x = {val};"
            ast = _c_parser.parse(code)
            init = ast.ext[0].init
            self.assertEqual(init.type, expected_type, f"Failed for {val}")


class TestCharacterConstantTypes(unittest.TestCase):
    def test_plain_char(self):
        ast = _c_parser.parse("char c = 'a';")
        self.assertEqual(ast.ext[0].init.type, "char")

    def test_wchar(self):
        ast = _c_parser.parse("typedef int wchar_t; wchar_t c = L'a';")
        self.assertEqual(ast.ext[1].init.type, "wchar_t")

    def test_u8char(self):
        ast = _c_parser.parse("char c = u8'a';")
        self.assertEqual(ast.ext[0].init.type, "char")

    def test_u16char(self):
        ast = _c_parser.parse("typedef unsigned short char16_t; char16_t c = u'a';")
        self.assertEqual(ast.ext[1].init.type, "char16_t")

    def test_u32char(self):
        ast = _c_parser.parse("typedef unsigned int char32_t; char32_t c = U'a';")
        self.assertEqual(ast.ext[1].init.type, "char32_t")


class TestStringConcatenation(unittest.TestCase):
    def test_plain_string_concat(self):
        ast = _c_parser.parse('char* s = "hello" " " "world";')
        self.assertEqual(ast.ext[0].init.value, '"hello world"')

    def test_empty_string_concat(self):
        ast = _c_parser.parse('char* s = "" "foo";')
        self.assertEqual(ast.ext[0].init.value, '"foo"')

    def test_wstring_concat_same_type(self):
        tests = [
            ('L"hello" L"world"', 'L"helloworld"'),
            ('u8"hello" u8"world"', 'u8"helloworld"'),
            ('u"hello" u"world"', 'u"helloworld"'),
            ('U"hello" U"world"', 'U"helloworld"'),
        ]
        for val, expected in tests:
            code = f"typedef int wchar_t; wchar_t* s = {val};"
            ast = _c_parser.parse(code)
            self.assertEqual(ast.ext[1].init.value, expected, f"Failed for {val}")

    def test_wstring_concat_mixed_fails(self):
        mixed = [
            'L"hello" u8"world"',
            'u"hello" U"world"',
            'u8"hello" L"world"',
        ]
        for val in mixed:
            code = f"typedef int wchar_t; wchar_t* s = {val};"
            with self.assertRaises(ParseError, msg=f"Should have failed for {val}"):
                _c_parser.parse(code)

    def test_three_strings_concat(self):
        ast = _c_parser.parse('char* s = "a" "b" "c";')
        self.assertEqual(ast.ext[0].init.value, '"abc"')

    def test_three_wstrings_concat(self):
        ast = _c_parser.parse('typedef int wchar_t; wchar_t* s = L"a" L"b" L"c";')
        self.assertEqual(ast.ext[1].init.value, 'L"abc"')


class TestKnRFunctionDefinitions(unittest.TestCase):
    def _parse_fdef(self, src):
        ast = _c_parser.parse(src)
        return ast.ext[0]

    def test_simple_kr_has_typed_params(self):
        f = self._parse_fdef("""
            int foo(a, b)
            int a;
            int b;
            {
                return a + b;
            }
        """)
        params = f.decl.type.args.params
        self.assertEqual(len(params), 2)
        self.assertIsInstance(params[0], c_ast.Decl)
        self.assertIsInstance(params[1], c_ast.Decl)
        self.assertEqual(params[0].name, "a")
        self.assertEqual(params[1].name, "b")

    def test_kr_array_param_decays_to_pointer(self):
        f = self._parse_fdef("""
            int foo(a)
            int a[5];
            {
                return a[0];
            }
        """)
        param = f.decl.type.args.params[0]
        self.assertIsInstance(param.type, c_ast.PtrDecl)
        self.assertIsInstance(f.param_decls[0].type, c_ast.ArrayDecl)

    def test_kr_function_pointer_param(self):
        f = self._parse_fdef("""
            int foo(g)
            int (*g)(int, float);
            {
                return g(1, 2.0);
            }
        """)
        param = f.decl.type.args.params[0]
        self.assertIsInstance(param.type, c_ast.PtrDecl)
        self.assertIsInstance(param.type.type, c_ast.FuncDecl)
        self.assertEqual(len(param.type.type.args.params), 2)

    def test_kr_pointer_param_stays_pointer(self):
        f = self._parse_fdef("""
            int foo(p)
            int *p;
            {
                return *p;
            }
        """)
        param = f.decl.type.args.params[0]
        self.assertIsInstance(param.type, c_ast.PtrDecl)

    def test_kr_no_param_decls_stays_id(self):
        f = self._parse_fdef("""
            foo(p)
            {
                return 3;
            }
        """)
        param = f.decl.type.args.params[0]
        self.assertIsInstance(param, c_ast.ID)

    def test_modern_function_not_affected(self):
        f = self._parse_fdef("""
            int foo(int a, float b) {
                return a + (int)b;
            }
        """)
        params = f.decl.type.args.params
        self.assertEqual(len(params), 2)
        self.assertIsInstance(params[0], c_ast.Decl)
        self.assertIsInstance(params[1], c_ast.Decl)
        self.assertIsNone(f.param_decls)


class TestSwitchCaseFlattening(unittest.TestCase):
    def test_empty_switch(self):
        code = """
            int main() {
                switch(x) {
                }
                return 0;
            }
        """
        ast = _c_parser.parse(code)
        switch = ast.ext[0].body.block_items[0]
        result = fix_switch_cases(switch)
        self.assertIsInstance(result.stmt, c_ast.Compound)
        self.assertEqual(len(result.stmt.block_items), 0)

    def test_case_with_empty_stmts_manual(self):
        case_node = c_ast.Case(c_ast.Constant("int", "1"), [])
        stmts_list = []
        from pycparser.ast_transforms import _extract_nested_case
        try:
            _extract_nested_case(case_node, stmts_list)
        except IndexError:
            self.fail("_extract_nested_case raised IndexError on empty stmts")

    def test_default_with_empty_stmts_manual(self):
        default_node = c_ast.Default([])
        stmts_list = []
        from pycparser.ast_transforms import _extract_nested_case
        try:
            _extract_nested_case(default_node, stmts_list)
        except IndexError:
            self.fail("_extract_nested_case raised IndexError on empty stmts")

    def test_default_with_nested_case(self):
        code = """
            int main() {
                switch(x) {
                    default:
                    case 1:
                        break;
                }
                return 0;
            }
        """
        ast = _c_parser.parse(code)
        switch = ast.ext[0].body.block_items[0]
        result = fix_switch_cases(switch)
        block_items = result.stmt.block_items
        self.assertEqual(len(block_items), 2)
        self.assertIsInstance(block_items[0], c_ast.Default)
        self.assertIsInstance(block_items[1], c_ast.Case)

    def test_nested_case_fallthrough(self):
        code = """
            int main() {
                switch(x) {
                    case 1:
                    case 2:
                        return 1;
                    default:
                        break;
                }
                return 0;
            }
        """
        ast = _c_parser.parse(code)
        switch = ast.ext[0].body.block_items[0]
        result = fix_switch_cases(switch)
        block_items = result.stmt.block_items
        self.assertEqual(len(block_items), 3)
        self.assertIsInstance(block_items[0], c_ast.Case)
        self.assertIsInstance(block_items[1], c_ast.Case)
        self.assertIsInstance(block_items[2], c_ast.Default)


class TestTypeChainCycleProtection(unittest.TestCase):
    def test_atomic_specifiers_no_infinite_loop(self):
        decl = c_ast.Decl(
            name="x",
            quals=[],
            align=None,
            storage=[],
            funcspec=[],
            type=c_ast.TypeDecl(
                declname="x",
                quals=[],
                align=None,
                type=c_ast.IdentifierType(["int"]),
            ),
            init=None,
            bitsize=None,
        )
        try:
            result = fix_atomic_specifiers(decl)
        except ValueError as e:
            self.fail(f"fix_atomic_specifiers raised ValueError: {e}")

    def test_type_chain_cycle_in_parser(self):
        code = "int x;"
        try:
            ast = _c_parser.parse(code)
        except Exception as e:
            self.fail(f"Parsing failed: {e}")

    def test_generator_depth_limit(self):
        ast = _c_parser.parse("int x;")
        try:
            _c_generator.visit(ast)
        except ValueError as e:
            self.fail(f"Generator raised ValueError: {e}")


class TestRoundTrip(unittest.TestCase):
    def test_modern_function(self):
        src = """
            int foo(int a, float b) {
                return a + (int)b;
            }
        """
        self.assertTrue(roundtrip_ok(src))

    def test_kr_simple_function(self):
        src = """
            int foo(a, b)
            int a;
            int b;
            {
                return a + b;
            }
        """
        self.assertTrue(roundtrip_ok(src))

    def test_kr_array_param(self):
        src = """
            int foo(a)
            int a[5];
            {
                return a[0];
            }
        """
        self.assertTrue(roundtrip_ok(src))

    def test_kr_function_pointer_param(self):
        src = """
            int foo(g)
            int (*g)(int, float);
            {
                return g(1, 2.0);
            }
        """
        self.assertTrue(roundtrip_ok(src))

    def test_complex_pointer_decl(self):
        src = "int *(*foo)(int, char*);"
        self.assertTrue(roundtrip_ok(src))

    def test_nested_switch_case(self):
        src = """
            int foo(int x) {
                switch(x) {
                    case 1:
                    case 2:
                        return 1;
                    default:
                        break;
                }
                return 0;
            }
        """
        self.assertTrue(roundtrip_ok(src))

    def test_bitfield_struct(self):
        src = """
            struct S {
                int a:3;
                int b:5;
                int c;
            };
        """
        self.assertTrue(roundtrip_ok(src))

    def test_multi_dimensional_array(self):
        src = "int arr[3][4][5];"
        self.assertTrue(roundtrip_ok(src))

    def test_enum(self):
        src = """
            enum Color {
                RED = 1,
                GREEN,
                BLUE = 5
            };
        """
        self.assertTrue(roundtrip_ok(src))

    def test_wide_string_concat(self):
        src = 'typedef int wchar_t; wchar_t* s = L"hello" L"world";'
        self.assertTrue(roundtrip_ok(src))

    def test_binary_literal_with_suffix(self):
        src = "unsigned long long a = 0b1010ull;"
        self.assertTrue(roundtrip_ok(src))

    def test_integer_suffixes(self):
        src = """
            unsigned long long a = 42ULL;
            long int b = 42L;
            unsigned int c = 42U;
        """
        self.assertTrue(roundtrip_ok(src))

    def test_float_suffixes(self):
        src = """
            float a = 1.0f;
            double b = 2.0;
            long double c = 3.0L;
        """
        self.assertTrue(roundtrip_ok(src))

    def test_string_concatenation(self):
        src = 'char* s = "hello" " " "world";'
        self.assertTrue(roundtrip_ok(src))


if __name__ == "__main__":
    unittest.main()
