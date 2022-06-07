use serde::{Deserialize, Serialize};

use super::b_field_element::BFieldElement;
use super::rescue_prime_params;
use super::stark::triton::table::aux_table;
use super::traits::PrimeField;

type Word = BFieldElement;

pub const RP_DEFAULT_OUTPUT_SIZE: usize = 5;
pub const RP_DEFAULT_WIDTH: usize = 16;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RescuePrimeXlix<const M: usize> {
    pub capacity: usize,
    pub n: usize,
    pub alpha: u64,
    pub alpha_inv: u64,
    pub mds: Vec<Vec<Word>>,
    pub mds_inv: Vec<Vec<Word>>,
    pub round_constants: Vec<Word>,
}

#[allow(clippy::needless_range_loop)]
impl<const M: usize> RescuePrimeXlix<M> {
    /// Hash an arbitrary-length slice of words into a vector of at most `M`
    /// words. Do this by feeding `rate` amount of words at a time to the
    /// `rescue_xlix_permutation`, leaving `capacity` number of words between
    /// iterations.
    pub fn hash(&self, input: &[Word], output_len: usize) -> Vec<Word> {
        assert!(output_len <= M, "Output at most {} words.", M);

        let rate = M - self.capacity;

        // Pad input
        let iterations = (input.len() / rate) + 1;
        let mut padded_input: Vec<Word> = vec![0.into(); iterations * rate];
        padded_input[0..input.len()].copy_from_slice(input);
        padded_input[input.len()] = 1.into();

        // Initialize state
        let mut state: [Word; M] = [0.into(); M];

        // Absorb
        for iteration in 0..iterations {
            let start = iteration * rate;
            let end = start + rate;
            let input_slice = &padded_input[start..end];
            state[0..rate].copy_from_slice(input_slice);

            self.rescue_xlix_permutation(&mut state);
        }

        // Squeeze
        (&state[0..output_len]).to_vec()
    }

    /// The Rescue-XLIX permutation
    pub fn rescue_xlix_permutation(&self, state: &mut [Word; M]) {
        debug_assert_eq!(M, state.len());

        for round in 0..self.n {
            self.rescue_xlix_round(round, state);
        }
    }

    pub fn rescue_xlix_permutation_trace(
        &self,
        state: &mut [Word; M],
    ) -> Vec<[Word; aux_table::BASE_WIDTH]> {
        debug_assert_eq!(M, state.len());
        let mut states: Vec<[Word; aux_table::BASE_WIDTH]> = Vec::with_capacity(self.n);

        let mut idc: Word = 0.into();
        let mut first_row = [idc; aux_table::BASE_WIDTH];
        first_row[1..].copy_from_slice(state);
        states.push(first_row);

        for round in 0..self.n {
            idc += 1.into();
            self.rescue_xlix_round(round, state);
            let mut row = [idc; aux_table::BASE_WIDTH];
            row[1..].copy_from_slice(state);
            states.push(row);
        }

        states
    }

    #[inline(always)]
    fn apply_inverse_sbox(state: &mut [Word; M]) {
        // Adapted from (MIT Licensed) https://github.com/novifinancial/winterfell/blob/main/math/src/field/f64/mod.rs#L40
        // compute base^10540996611094048183 using 72 multiplications per array element
        // 10540996611094048183 = b1001001001001001001001001001000110110110110110110110110110110111

        // compute base^10
        let mut pow_1 = *state;
        pow_1.iter_mut().for_each(|p| *p *= *p);

        // compute base^100
        let mut pow_2 = pow_1;
        pow_2.iter_mut().for_each(|p| *p *= *p);

        // compute base^100100
        let pow_3 = BFieldElement::power_accumulator::<M, 3>(pow_2, pow_2);

        // compute base^100100100100
        let pow_4 = BFieldElement::power_accumulator::<M, 6>(pow_3, pow_3);

        // compute base^100100100100100100100100
        let pow_5 = BFieldElement::power_accumulator::<M, 12>(pow_4, pow_4);

        // compute base^100100100100100100100100100100
        let pow_6 = BFieldElement::power_accumulator::<M, 6>(pow_5, pow_3);

        // compute base^1001001001001001001001001001000100100100100100100100100100100
        let pow_7 = BFieldElement::power_accumulator::<M, 31>(pow_6, pow_6);

        // compute base^1001001001001001001001001001000110110110110110110110110110110111
        for (i, s) in state.iter_mut().enumerate() {
            let a = (pow_7[i].square() * pow_6[i]).square().square();
            let b = pow_1[i] * pow_2[i] * *s;
            *s = a * b;
        }
    }

    #[inline]
    fn rescue_xlix_round(&self, i: usize, state: &mut [Word; M]) {
        // S-box
        for j in 0..M {
            state[j] = state[j].mod_pow(self.alpha);
        }

        // MDS
        for j in 0..M {
            for k in 0..M {
                state[j] += MDS[j][k] * state[k];
            }
        }

        // Round constants
        for j in 0..M {
            state[j] += self.round_constants[i * 2 * M + j];
        }

        // Inverse S-box
        Self::apply_inverse_sbox(state);

        // MDS
        for j in 0..M {
            for k in 0..M {
                state[j] += MDS[j][k] * state[k];
            }
        }

        // Round constants
        for j in 0..M {
            state[j] += self.round_constants[i * 2 * M + M + j];
        }
    }
}

pub fn neptune_params() -> RescuePrimeXlix<RP_DEFAULT_WIDTH> {
    let params = rescue_prime_params::rescue_prime_params_bfield_0();

    let capacity = 4;
    let n = params.steps_count;
    let alpha = params.alpha;
    let alpha_inv = params.alpha_inv;
    let mds = params.mds;
    let mds_inv = params.mds_inv;
    let round_constants = params.round_constants;

    RescuePrimeXlix {
        capacity,
        n,
        alpha,
        alpha_inv,
        mds,
        mds_inv,
        round_constants,
    }
}

#[cfg(test)]
mod rescue_prime_xlix_tests {
    use super::*;

    #[test]
    fn test_vector_0() {
        let rp = neptune_params();
        let input: Vec<BFieldElement> = vec![BFieldElement::ring_one()];
        let expected = vec![
            BFieldElement::new(15634820042269645118),
            BFieldElement::new(615341424773519402),
            BFieldElement::new(7368749134254585916),
            BFieldElement::new(6434330208930178748),
            BFieldElement::new(7150561627751137065),
        ];
        assert_eq!(expected, rp.hash(&input, 5));
    }

    #[test]
    fn test_vector_1() {
        let rp = neptune_params();
        let start_value = 74620000171;
        let width = 10;
        let input: Vec<BFieldElement> = (start_value..start_value + width)
            .into_iter()
            .map(BFieldElement::new)
            .collect::<Vec<_>>();
        let expected = vec![
            BFieldElement::new(18207611346694155661),
            BFieldElement::new(5358489668086158029),
            BFieldElement::new(15218675170619297004),
            BFieldElement::new(12919464649779078983),
            BFieldElement::new(9284515517624112714),
        ];
        assert_eq!(expected, rp.hash(&input, 5));
    }

    #[test]
    fn test_vector_2() {
        let rp = neptune_params();
        let start_value = 52;
        let width = 12;
        let input: Vec<BFieldElement> = (start_value..start_value + width)
            .into_iter()
            .map(BFieldElement::new)
            .collect::<Vec<_>>();
        let expected = vec![
            BFieldElement::new(13854889922040347713),
            BFieldElement::new(9107023863351443899),
            BFieldElement::new(10066861977733156370),
            BFieldElement::new(12168766094991429332),
            BFieldElement::new(14235729488804827283),
        ];
        assert_eq!(expected, rp.hash(&input, 5));
    }

    #[test]
    fn test_vector_3() {
        let rp = neptune_params();
        let start_value = 1000;
        let width = 37;
        let input: Vec<BFieldElement> = (start_value..start_value + width)
            .into_iter()
            .map(BFieldElement::new)
            .collect::<Vec<_>>();
        let expected = vec![
            BFieldElement::new(6860796620242995210),
            BFieldElement::new(5250556310206967797),
            BFieldElement::new(15566964441017148761),
            BFieldElement::new(7948069663969379846),
            BFieldElement::new(10491176845836052780),
        ];
        assert_eq!(expected, rp.hash(&input, 5));
    }
}

const MDS: [[BFieldElement; RP_DEFAULT_WIDTH]; RP_DEFAULT_WIDTH] = [
    [
        BFieldElement::new(5910257123858819639),
        BFieldElement::new(3449115226714951713),
        BFieldElement::new(16770055338049327985),
        BFieldElement::new(610399731775780810),
        BFieldElement::new(7363016345531076300),
        BFieldElement::new(16174724756564259629),
        BFieldElement::new(8736587794472183152),
        BFieldElement::new(12699016954477470956),
        BFieldElement::new(13948112026909862966),
        BFieldElement::new(18015813124076612987),
        BFieldElement::new(9568929147539067610),
        BFieldElement::new(14859461777592116402),
        BFieldElement::new(18169364738825153183),
        BFieldElement::new(18221568702798258352),
        BFieldElement::new(1524268296724555606),
        BFieldElement::new(5538821761600),
    ],
    [
        BFieldElement::new(1649528676200182784),
        BFieldElement::new(336497118937017052),
        BFieldElement::new(15805000027048028625),
        BFieldElement::new(15709375513998678646),
        BFieldElement::new(14837031240173858084),
        BFieldElement::new(11366298206428370494),
        BFieldElement::new(15698532768527519720),
        BFieldElement::new(5911577595727321095),
        BFieldElement::new(16676030327621016157),
        BFieldElement::new(16537624251746851423),
        BFieldElement::new(13325141695736654367),
        BFieldElement::new(9337952653454313447),
        BFieldElement::new(9090375522091353302),
        BFieldElement::new(5605636660979522224),
        BFieldElement::new(6357222834896114791),
        BFieldElement::new(7776871531164456679),
    ],
    [
        BFieldElement::new(8264739868177574620),
        BFieldElement::new(12732288338686680125),
        BFieldElement::new(13022293791945187811),
        BFieldElement::new(17403057736098613442),
        BFieldElement::new(2871266924987061743),
        BFieldElement::new(13286707530570640459),
        BFieldElement::new(9229362695439112266),
        BFieldElement::new(815317759014579856),
        BFieldElement::new(7447771153889267897),
        BFieldElement::new(2209002535000750347),
        BFieldElement::new(3280506473249596174),
        BFieldElement::new(13756142018694965622),
        BFieldElement::new(10518080861296830621),
        BFieldElement::new(16578355848983066277),
        BFieldElement::new(12732532221704648123),
        BFieldElement::new(3426526797578099186),
    ],
    [
        BFieldElement::new(8563516248221808333),
        BFieldElement::new(13079317959606236131),
        BFieldElement::new(15645458946300428515),
        BFieldElement::new(9958819147895829140),
        BFieldElement::new(13028053188247480206),
        BFieldElement::new(6789511720078828478),
        BFieldElement::new(6583246594815170294),
        BFieldElement::new(4423695887326249884),
        BFieldElement::new(9751139665897711642),
        BFieldElement::new(10039202025292797758),
        BFieldElement::new(12208726994829996150),
        BFieldElement::new(6238795140281096003),
        BFieldElement::new(9113696057226188857),
        BFieldElement::new(9898705245385052191),
        BFieldElement::new(4213712701625520075),
        BFieldElement::new(8038355032286280912),
    ],
    [
        BFieldElement::new(426685147605824917),
        BFieldElement::new(7673465577918025498),
        BFieldElement::new(8452867379070564008),
        BFieldElement::new(10827610229277395180),
        BFieldElement::new(16155539332955658546),
        BFieldElement::new(1575428636717115288),
        BFieldElement::new(8765972548498757598),
        BFieldElement::new(8405996249707890526),
        BFieldElement::new(14855028677418679455),
        BFieldElement::new(17878170012428694685),
        BFieldElement::new(16572621079016066883),
        BFieldElement::new(5311046098447994501),
        BFieldElement::new(10635376800783355348),
        BFieldElement::new(14205668690430323921),
        BFieldElement::new(1181422971831412672),
        BFieldElement::new(4651053123208915543),
    ],
    [
        BFieldElement::new(12465667489477238576),
        BFieldElement::new(7300129031676503132),
        BFieldElement::new(13458544786180633209),
        BFieldElement::new(8946801771555977477),
        BFieldElement::new(14203890406114400141),
        BFieldElement::new(8219081892380458635),
        BFieldElement::new(6035067543134909245),
        BFieldElement::new(15140374581570897616),
        BFieldElement::new(4514006299509426029),
        BFieldElement::new(16757530089801321524),
        BFieldElement::new(13202061911440346802),
        BFieldElement::new(11227558237427129334),
        BFieldElement::new(315998614524336401),
        BFieldElement::new(11280705904396606227),
        BFieldElement::new(5798516367202621128),
        BFieldElement::new(17154761698338453414),
    ],
    [
        BFieldElement::new(13574436947400004837),
        BFieldElement::new(3126509266905053998),
        BFieldElement::new(10740979484255925394),
        BFieldElement::new(9273322683773825324),
        BFieldElement::new(15349096509718845737),
        BFieldElement::new(14694022445619674948),
        BFieldElement::new(8733857890739087596),
        BFieldElement::new(3198488337424282101),
        BFieldElement::new(9521016570828679381),
        BFieldElement::new(11267736037298472148),
        BFieldElement::new(14825280481028844943),
        BFieldElement::new(1326588754335738002),
        BFieldElement::new(6200834522767914499),
        BFieldElement::new(1070210996042416038),
        BFieldElement::new(9140190343656907671),
        BFieldElement::new(15531381283521001952),
    ],
    [
        BFieldElement::new(253143295675927354),
        BFieldElement::new(11977331414401291539),
        BFieldElement::new(13941376566367813256),
        BFieldElement::new(469904915148256197),
        BFieldElement::new(10873951860155749104),
        BFieldElement::new(3939719938926157877),
        BFieldElement::new(2271392376641547055),
        BFieldElement::new(4725974756185387075),
        BFieldElement::new(14827835543640648161),
        BFieldElement::new(17663273767033351157),
        BFieldElement::new(12440960700789890843),
        BFieldElement::new(16589620022628590428),
        BFieldElement::new(12838889473653138505),
        BFieldElement::new(11170336581460183657),
        BFieldElement::new(7583333056198317221),
        BFieldElement::new(6006908286410425140),
    ],
    [
        BFieldElement::new(15648567098514276013),
        BFieldElement::new(188901633101859949),
        BFieldElement::new(12256163716419861419),
        BFieldElement::new(17319784688409668747),
        BFieldElement::new(9648971065289440425),
        BFieldElement::new(11370683735445551679),
        BFieldElement::new(11265203235776280908),
        BFieldElement::new(1737672785338087677),
        BFieldElement::new(5225587291780939578),
        BFieldElement::new(4739055740469849012),
        BFieldElement::new(1212344601223444182),
        BFieldElement::new(12958616893209019599),
        BFieldElement::new(7922060480554370635),
        BFieldElement::new(14661420107595710445),
        BFieldElement::new(11744359917257111592),
        BFieldElement::new(9674559564931202709),
    ],
    [
        BFieldElement::new(8326110231976411065),
        BFieldElement::new(16856751238353701757),
        BFieldElement::new(7515652322254196544),
        BFieldElement::new(2062531989536141174),
        BFieldElement::new(3875321171362100965),
        BFieldElement::new(1164854003752487518),
        BFieldElement::new(3997098993859160292),
        BFieldElement::new(4074090397542250057),
        BFieldElement::new(3050858158567944540),
        BFieldElement::new(4568245569065883863),
        BFieldElement::new(14559440781022773799),
        BFieldElement::new(5401845794552358815),
        BFieldElement::new(6544584366002554176),
        BFieldElement::new(2511522072283652847),
        BFieldElement::new(9759884967674698659),
        BFieldElement::new(16411672358681189856),
    ],
    [
        BFieldElement::new(11392578809073737776),
        BFieldElement::new(8013631514034873271),
        BFieldElement::new(11439549174997471674),
        BFieldElement::new(6373021446442411366),
        BFieldElement::new(12491600135569477757),
        BFieldElement::new(1017093281401495736),
        BFieldElement::new(663547836518863091),
        BFieldElement::new(16157302719777897692),
        BFieldElement::new(11208801522915446640),
        BFieldElement::new(10058178191286215107),
        BFieldElement::new(5521712058210208094),
        BFieldElement::new(3611681474253815005),
        BFieldElement::new(4864578569041337696),
        BFieldElement::new(12270319000993569289),
        BFieldElement::new(7347066511426336318),
        BFieldElement::new(6696546239958933736),
    ],
    [
        BFieldElement::new(3335469193383486908),
        BFieldElement::new(12719366334180058014),
        BFieldElement::new(14123019207894489639),
        BFieldElement::new(11418186023060178542),
        BFieldElement::new(2042199956854124583),
        BFieldElement::new(17539253100488345226),
        BFieldElement::new(16240833881391672847),
        BFieldElement::new(11712520063241304909),
        BFieldElement::new(6456900719511754234),
        BFieldElement::new(1819022137223501306),
        BFieldElement::new(7371152900053879920),
        BFieldElement::new(6521878675261223812),
        BFieldElement::new(2050999666988944811),
        BFieldElement::new(8262038465464898064),
        BFieldElement::new(13303819303390508091),
        BFieldElement::new(12657292926928303663),
    ],
    [
        BFieldElement::new(8794128680724662595),
        BFieldElement::new(4068577832515945116),
        BFieldElement::new(758247715040138478),
        BFieldElement::new(5600369601992438532),
        BFieldElement::new(3369463178350382224),
        BFieldElement::new(13763645328734311418),
        BFieldElement::new(9685701761982837416),
        BFieldElement::new(2711119809520557835),
        BFieldElement::new(11680482056777716424),
        BFieldElement::new(10958223503056770518),
        BFieldElement::new(4168390070510137163),
        BFieldElement::new(10823375744683484459),
        BFieldElement::new(5613197991565754677),
        BFieldElement::new(11781942063118564684),
        BFieldElement::new(9352512500813609723),
        BFieldElement::new(15997830646514778986),
    ],
    [
        BFieldElement::new(7407352006524266457),
        BFieldElement::new(15312663387608602775),
        BFieldElement::new(3026364159907661789),
        BFieldElement::new(5698531403379362946),
        BFieldElement::new(2544271242593770624),
        BFieldElement::new(13104502948897878458),
        BFieldElement::new(7840062700088318710),
        BFieldElement::new(6028743588538970215),
        BFieldElement::new(6144415809411296980),
        BFieldElement::new(468368941216390216),
        BFieldElement::new(3638618405705274008),
        BFieldElement::new(11105401941482704573),
        BFieldElement::new(1850274872877725129),
        BFieldElement::new(1011155312563349004),
        BFieldElement::new(3234620948537841909),
        BFieldElement::new(3818372677739507813),
    ],
    [
        BFieldElement::new(4863130691592118581),
        BFieldElement::new(8942166964590283171),
        BFieldElement::new(3639677194051371072),
        BFieldElement::new(15477372418124081864),
        BFieldElement::new(10322228711752830209),
        BFieldElement::new(9139111778956611066),
        BFieldElement::new(202171733050704358),
        BFieldElement::new(11982413146686512577),
        BFieldElement::new(11001000478006340870),
        BFieldElement::new(5491471715020327065),
        BFieldElement::new(6969114856449768266),
        BFieldElement::new(11088492421847219924),
        BFieldElement::new(12913509272810999025),
        BFieldElement::new(17366506887360149369),
        BFieldElement::new(7036328554328346102),
        BFieldElement::new(11139255730689011050),
    ],
    [
        BFieldElement::new(2844974929907956457),
        BFieldElement::new(6488525141985913483),
        BFieldElement::new(2860098796699131680),
        BFieldElement::new(10366343151884073105),
        BFieldElement::new(844875652557703984),
        BFieldElement::new(1053177270393416978),
        BFieldElement::new(5189466196833763142),
        BFieldElement::new(1024738234713107670),
        BFieldElement::new(8846741799369572841),
        BFieldElement::new(14490406830213564822),
        BFieldElement::new(10577371742628912722),
        BFieldElement::new(3276210642025060502),
        BFieldElement::new(2605621719516949928),
        BFieldElement::new(5417148926702080639),
        BFieldElement::new(11100652475866543814),
        BFieldElement::new(5247366835775169839),
    ],
];
